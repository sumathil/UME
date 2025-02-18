/*
  Copyright (c) 2023, Triad National Security, LLC. All rights reserved.

  This is open source software; you can redistribute it and/or modify it under
  the terms of the BSD-3 License. If software is modified to produce derivative
  works, such modified software should be clearly marked, so as not to confuse
  it with the version available from LANL. Full text of the BSD-3 License can be
  found in the LICENSE.md file, and the full assertion of copyright in the
  NOTICE.md file.
*/

/*!
  \file Ume/gradient.cc
*/
#include <Kokkos_Core.hpp>
#include <iostream>
#include <omp.h>
#include "Ume/gradient.hh"
#include "Ume/Timer.hh"
#include "Ume/utils.hh"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace Ume {

using DBLV_T = DS_Types::DBLV_T;
using VEC3V_T = DS_Types::VEC3V_T;
using VEC3_T = DS_Types::VEC3_T;

void gradzatp(Ume::SOA_Idx::Mesh &mesh, DBLV_T const &zone_field,
    VEC3V_T &point_gradient) {
  auto const &csurf = mesh.ds->caccess_vec3v("corner_csurf");
  auto const &corner_volume = mesh.ds->caccess_dblv("corner_vol");
  auto const &point_normal = mesh.ds->caccess_vec3v("point_norm");
  auto const &c_to_p_map = mesh.ds->caccess_intv("m:c>p");
  auto const &c_to_z_map = mesh.ds->caccess_intv("m:c>z");
  auto const &corner_type = mesh.corners.mask;
  auto const &point_type = mesh.points.mask;

  int const pll = mesh.points.size();
  int const pl = mesh.points.local_size();
  int const cl = mesh.corners.local_size();

  DBLV_T point_volume(pll, 0.0);
  point_gradient.assign(pll, VEC3_T(0.0));

#define KOKKOS_SPACE Kokkos::HostSpace  
Ume::Timer zatp_time;
zatp_time.start();

Kokkos::View<const Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_point_normal(&point_normal[0], point_normal.size());
Kokkos::View<const int *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > k_c_to_z_map(&c_to_z_map[0], c_to_z_map.size());
Kokkos::View<const int *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > k_c_to_p_map(&c_to_p_map[0], c_to_p_map.size());
Kokkos::View<const double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_corner_volume(&corner_volume[0], corner_volume.size());
Kokkos::View<double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_point_volume(&point_volume[0], point_volume.size());
Kokkos::View<Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_point_gradient(&point_gradient[0], point_gradient.size());
Kokkos::View<const Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_csurf(&csurf[0], csurf.size());
Kokkos::View<const double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_zone_field(&zone_field[0], zone_field.size());
Kokkos::View<const short *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_point_type(&point_type[0], point_type.size());
Kokkos::View<const short *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_corner_type(&corner_type[0], corner_type.size());

  Kokkos::parallel_for("gradzatp-1", cl, KOKKOS_LAMBDA (const int c) {
    if (k_corner_type[c] >= 1) {  
      int const z = k_c_to_z_map[c];
      int const p = k_c_to_p_map[c];
      k_point_volume[p] += k_corner_volume[c];
      k_point_gradient[p] += k_csurf[c] * k_zone_field[z];
    }
  });

  mesh.points.gathscat(Ume::Comm::Op::SUM, point_volume);
  mesh.points.gathscat(Ume::Comm::Op::SUM, point_gradient);

  /*
    Divide by point control volume to get gradient.  If a point is on the outer
    perimeter of the mesh (POINT_TYPE=-1), subtract the outward normal component
    of the gradient using the point normals.
   */
  Kokkos::parallel_for("gradzatp-2", pl, KOKKOS_LAMBDA (const int p) {
    if (k_point_type[p] > 0) {
      // Internal point
      k_point_gradient[p] /= k_point_volume[p];
    } else if (k_point_type[p] == -1) {
      // Mesh boundary point
      double const ppdot = dotprod(k_point_gradient[p], k_point_normal[p]);
      k_point_gradient[p] = (k_point_gradient[p] - k_point_normal[p] * ppdot) / k_point_volume[p];
    }
  });
zatp_time.stop();
std::cout << "zatp_time: " << zatp_time.seconds() << "s\n";

  mesh.points.scatter(point_gradient);
}

void gradzatz(Ume::SOA_Idx::Mesh &mesh, DBLV_T const &zone_field,
    VEC3V_T &zone_gradient, VEC3V_T &point_gradient) {
  auto const &c_to_z_map = mesh.ds->caccess_intv("m:c>z");
  auto const &c_to_p_map = mesh.ds->caccess_intv("m:c>p");
  int const num_local_corners = mesh.corners.local_size();
  auto const &corner_type = mesh.corners.mask;
  auto const &corner_volume = mesh.ds->caccess_dblv("corner_vol");

  // Get the field gradient at each mesh point.
  gradzatp(mesh, zone_field, point_gradient);
  /* Accumulate the zone volume.  Note that we need to allocate a zone field for
     volume, as we are accumulating from corners */
  DBLV_T zone_volume(mesh.zones.size(), 0.0);

#define KOKKOS_SPACE Kokkos::HostSpace
Ume::Timer zatz_time;
zatz_time.start();

Kokkos::View<const int *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > k_c_to_z_map(&c_to_z_map[0], c_to_z_map.size());
Kokkos::View<const int *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > k_c_to_p_map(&c_to_p_map[0], c_to_p_map.size());
Kokkos::View<const double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_corner_volume(&corner_volume[0], corner_volume.size());
Kokkos::View<double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_zone_volume(&zone_volume[0], zone_volume.size());
Kokkos::View<Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_point_gradient(&point_gradient[0], point_gradient.size());
Kokkos::View<const short *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_corner_type(&corner_type[0], corner_type.size());

  Kokkos::parallel_for("gradzatp-ivt", num_local_corners, KOKKOS_LAMBDA (const int corner_idx) {
    if (k_corner_type[corner_idx] >= 1){
      int const zone_idx = k_c_to_z_map[corner_idx];
      /* Note that we cannot parallelize across corners, as multiple corners
       write to the same zone. */
      k_zone_volume[zone_idx] += k_corner_volume[corner_idx];
    }
  });

  // Accumulate the zone-centered gradient
  zone_gradient.assign(mesh.zones.size(), VEC3_T(0.0));
  Kokkos::View<Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_zone_gradient(&zone_gradient[0], zone_gradient.size());
  Kokkos::parallel_for("gradzatp-ivt", num_local_corners, KOKKOS_LAMBDA (const int corner_idx) {
    if (k_corner_type[corner_idx] >= 1){
      int const zone_idx = k_c_to_z_map[corner_idx];
      int const point_idx = k_c_to_p_map[corner_idx];
      double const c_z_vol_ratio =
      k_corner_volume[corner_idx] / k_zone_volume[zone_idx];
      k_zone_gradient[zone_idx] += k_point_gradient[point_idx] * c_z_vol_ratio;
     }
    });
zatz_time.stop();
std::cout << "zatz_time: " << zatz_time.seconds() << "s\n";

  mesh.zones.scatter(zone_gradient);
}

void gradzatp_invert(Ume::SOA_Idx::Mesh &mesh, DBLV_T const &zone_field,
    VEC3V_T &point_gradient) {
  auto const &csurf = mesh.ds->caccess_vec3v("corner_csurf");
  auto const &corner_volume = mesh.ds->caccess_dblv("corner_vol");
  auto const &point_normal = mesh.ds->caccess_vec3v("point_norm");
  auto const &p_to_c_map = mesh.ds->caccess_intrr("m:p>rc");
  auto const &c_to_z_map = mesh.ds->caccess_intv("m:c>z");
  auto const &point_type = mesh.points.mask;

  int const num_points = mesh.points.size();
  int const num_local_points = mesh.points.local_size();

  DBLV_T point_volume(num_points, 0.0);
  point_gradient.assign(num_points, VEC3_T(0.0));

#define KOKKOS_SPACE Kokkos::HostSpace
Ume::Timer zatpivt_time;
zatpivt_time.start();

Kokkos::View<const int *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > k_c_to_z_map(&c_to_z_map[0], c_to_z_map.size());
Kokkos::View<const double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_corner_volume(&corner_volume[0], corner_volume.size());
Kokkos::View<double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_point_volume(&point_volume[0], point_volume.size());
Kokkos::View<Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_point_gradient(&point_gradient[0], point_gradient.size());
Kokkos::View<const Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_csurf(&csurf[0], csurf.size());
Kokkos::View<const double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_zone_field(&zone_field[0], zone_field.size());
Kokkos::View<const Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_point_normal(&point_normal[0], point_normal.size());
Kokkos::View<const short *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_point_type(&point_type[0], point_type.size());

  Kokkos::parallel_for("gradzatp-ivt", num_local_points, KOKKOS_LAMBDA (const int point_idx) {
    for (int const &corner_idx : p_to_c_map[point_idx]) {
      int const zone_idx = k_c_to_z_map[corner_idx];
      k_point_volume[point_idx] += k_corner_volume[corner_idx];
      k_point_gradient[point_idx] += k_csurf[corner_idx] * k_zone_field[zone_idx];
    }
  });

   Kokkos::parallel_for("gradzatp-ivt-2", num_local_points, KOKKOS_LAMBDA (const int point_idx) {
    if (k_point_type[point_idx] > 0) {
      // Internal point
      k_point_gradient[point_idx] /= k_point_volume[point_idx];
    } else if (k_point_type[point_idx] == -1) {
      // Mesh boundary point
      double const ppdot =
          dotprod(k_point_gradient[point_idx], k_point_normal[point_idx]);
      k_point_gradient[point_idx] = (k_point_gradient[point_idx] - k_point_normal[point_idx] * ppdot) /
          k_point_volume[point_idx];
    }
  });

zatpivt_time.stop();
std::cout << "zatpivt_time: " << zatpivt_time.seconds() << "s\n";
 
  mesh.points.scatter(point_gradient);
}

void gradzatz_invert(Ume::SOA_Idx::Mesh &mesh, DBLV_T const &zone_field,
    VEC3V_T &zone_gradient, VEC3V_T &point_gradient) {
  auto const &z_to_c_map = mesh.ds->caccess_intrr("m:z>c");
  auto const &c_to_p_map = mesh.ds->caccess_intv("m:c>p");
  int const num_local_zones = mesh.zones.local_size();
  auto const &zone_type = mesh.zones.mask;
  auto const &corner_volume = mesh.ds->caccess_dblv("corner_vol");

  // Get the field gradient at each mesh point.
  gradzatp_invert(mesh, zone_field, point_gradient);
  zone_gradient.assign(mesh.zones.size(), VEC3_T(0.0));

Ume::Timer zatzivt_time;
zatzivt_time.start();

Kokkos::View<const short *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_zone_type(&zone_type[0], zone_type.size());
Kokkos::View<const double *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  k_corner_volume(&corner_volume[0], corner_volume.size());
Kokkos::View<const int *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > k_c_to_p_map(&c_to_p_map[0], c_to_p_map.size());
Kokkos::View<Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_zone_gradient(&zone_gradient[0], zone_gradient.size());
Kokkos::View<Vec3 *, KOKKOS_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> k_point_gradient(&point_gradient[0], point_gradient.size());

  Kokkos::parallel_for("gradzatz-ivt-1", num_local_zones, KOKKOS_LAMBDA (const int zone_idx) {
    if (k_zone_type[zone_idx] >= 1){
      // Only operate on local interior zones

      // Accumulate the (local) zone volume
      double zone_volume{0.0}; // Only need a local volume
      for (int const &corner_idx : z_to_c_map[zone_idx]) {
        zone_volume += k_corner_volume[corner_idx];
      }

      for (int const &corner_idx : z_to_c_map[zone_idx]) {
        int const point_idx = k_c_to_p_map[corner_idx];
        double const c_z_vol_ratio = k_corner_volume[corner_idx] / zone_volume;
        k_zone_gradient[zone_idx] += k_point_gradient[point_idx] * c_z_vol_ratio;
      }
    }
  });

zatzivt_time.stop();
std::cout << "zatzivt_time: " << zatzivt_time.seconds() << "s\n";

  mesh.zones.scatter(zone_gradient);
}

} // namespace Ume
