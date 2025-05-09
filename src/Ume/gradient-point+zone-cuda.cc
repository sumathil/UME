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
#include "Ume/gradient.hh"

#include <iostream>
#include <Kokkos_Core.hpp>

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

  for (int c = 0; c < cl; ++c) {
    if (corner_type[c] < 1)
      continue; // Only operate on interior corners
    int const z = c_to_z_map[c];
    int const p = c_to_p_map[c];
    point_volume[p] += corner_volume[c];
    point_gradient[p] += csurf[c] * zone_field[z];
  }

  mesh.points.gathscat(Ume::Comm::Op::SUM, point_volume);
  mesh.points.gathscat(Ume::Comm::Op::SUM, point_gradient);

  /*
    Divide by point control volume to get gradient.  If a point is on the outer
    perimeter of the mesh (POINT_TYPE=-1), subtract the outward normal component
    of the gradient using the point normals.
   */
  for (int p = 0; p < pl; ++p) {
    if (point_type[p] > 0) {
      // Internal point
      point_gradient[p] /= point_volume[p];
    } else if (point_type[p] == -1) {
      // Mesh boundary point
      double const ppdot = dotprod(point_gradient[p], point_normal[p]);
      point_gradient[p] =
          (point_gradient[p] - point_normal[p] * ppdot) / point_volume[p];
    }
  }
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
  //    gradzatp(mesh, zone_field, point_gradient);
  // START gradzatp

  auto const &csurf = mesh.ds->caccess_vec3v("corner_csurf");
  auto const &point_normal = mesh.ds->caccess_vec3v("point_norm");
  auto const &point_type = mesh.points.mask;

  int const pll = mesh.points.size();
  int const pl = mesh.points.local_size();
  int const cl = mesh.corners.local_size();

  DBLV_T point_volume(pll, 0.0);
  point_gradient.assign(pll, VEC3_T(0.0));


#define HOST_SPACE Kokkos::HostSpace

  using space_t = Kokkos::DefaultExecutionSpace::memory_space;

  //Kokkos host views
  Kokkos::View<Vec3 *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Atomic>> h_point_gradient(&point_gradient[0], point_gradient.size());
  Kokkos::View<const short *, HOST_SPACE>  h_point_type(&point_type[0], point_type.size());
  Kokkos::View<double *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Atomic>>  h_point_volume(&point_volume[0], point_volume.size());
  Kokkos::View<const Vec3 *, HOST_SPACE>  h_point_normal(&point_normal[0], point_normal.size());

  Kokkos::View<const int *, HOST_SPACE> h_c_to_z_map(&c_to_z_map[0], c_to_z_map.size());
  Kokkos::View<const Vec3 *, HOST_SPACE>  h_csurf(&csurf[0], csurf.size());
  Kokkos::View<const double *, HOST_SPACE>  h_corner_volume(&corner_volume[0], corner_volume.size());
  Kokkos::View<const double *, HOST_SPACE>  h_zone_field(&zone_field[0], zone_field.size());

  Kokkos::View<const int *, HOST_SPACE> h_c_to_p_map(&c_to_p_map[0], c_to_p_map.size());
  Kokkos::View<const short *, HOST_SPACE>  h_corner_type(&corner_type[0], corner_type.size());

  // Kokkos mirror views
  auto d_point_gradient = create_mirror_view ( space_t (), h_point_gradient );
  auto d_point_type = create_mirror_view ( space_t () , h_point_type );
  auto d_point_volume = create_mirror_view ( space_t () , h_point_volume );
  auto d_point_normal = create_mirror_view ( space_t () , h_point_normal );

  auto d_c_to_z_map = create_mirror_view ( space_t () , h_c_to_z_map );
  auto d_csurf = create_mirror_view ( space_t () , h_csurf );
  auto d_corner_volume = create_mirror_view ( space_t () , h_corner_volume );
  auto d_zone_field = create_mirror_view ( space_t () , h_zone_field );

  auto d_c_to_p_map = create_mirror_view ( space_t () , h_c_to_p_map );
  auto d_corner_type = create_mirror_view ( space_t () , h_corner_type );

    // copy host to device
  Kokkos::deep_copy(d_point_gradient, h_point_gradient);
  Kokkos::deep_copy(d_point_type, h_point_type);
  Kokkos::deep_copy(d_point_volume, h_point_volume);
  Kokkos::deep_copy(d_point_normal, h_point_normal);
  Kokkos::deep_copy(d_c_to_z_map, h_c_to_z_map);
  Kokkos::deep_copy(d_csurf, h_csurf);
  Kokkos::deep_copy(d_corner_volume, h_corner_volume);
  Kokkos::deep_copy(d_zone_field, h_zone_field);
  Kokkos::deep_copy(d_c_to_p_map, h_c_to_p_map);
  Kokkos::deep_copy(d_corner_type, h_corner_type);
 

  // calculate on device
  Kokkos::parallel_for("gradzatz-1-cuda", cl, KOKKOS_LAMBDA (const int c) {
    if (d_corner_type[c] >= 1){  
      // Only operate on interior corners
      int const z = d_c_to_z_map[c];
      int const p = d_c_to_p_map[c];
      // the following two should be atomic operations
      d_point_volume[p] += d_corner_volume[c];
      d_point_gradient[p] += d_csurf[c] * d_zone_field[z];
    }
  });

  Kokkos::fence();

  /*
    Divide by point control volume to get gradient.  If a point is on the outer
    perimeter of the mesh (POINT_TYPE=-1), subtract the outward normal component
    of the gradient using the point normals.
   */
  Kokkos::parallel_for("gradzatz-2-cuda", pl, KOKKOS_LAMBDA (const int p) {
  if (d_point_type[p] > 0) {
      // Internal point
      d_point_gradient[p] /= d_point_volume[p];
    } else if (d_point_type[p] == -1) {
      // Mesh boundary point
      double const ppdot = dotprod(d_point_gradient[p], d_point_normal[p]);
      d_point_gradient[p] =
          (d_point_gradient[p] - d_point_normal[p] * ppdot) / d_point_volume[p];
    }
  });

  Kokkos::fence();

  // END gradzatp

  /* Accumulate the zone volume.  Note that we need to allocate a zone field for
     volume, as we are accumulating from corners */
  DBLV_T zone_volume(mesh.zones.size(), 0.0);

  Kokkos::View<double *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Atomic>>  h_zone_volume(&zone_volume[0], zone_volume.size());
  auto d_zone_volume = create_mirror_view ( space_t () , h_zone_volume );
  Kokkos::deep_copy(d_zone_volume, h_zone_volume);


  Kokkos::parallel_for("gradzatz-2-cuda", num_local_corners, KOKKOS_LAMBDA (const int corner_idx) {
    if (d_corner_type[corner_idx] >= 1){ 
    // Only operate on interior corners
    int const zone_idx = d_c_to_z_map[corner_idx];
    /* Note that we cannot parallelize across corners, as multiple corners
       write to the same zone. */
       // following should be atomic operation -- create 'atomic' mirror view
          d_zone_volume[zone_idx] += d_corner_volume[corner_idx];
    }
  });

  Kokkos::fence();

  // Accumulate the zone-centered gradient
  zone_gradient.assign(mesh.zones.size(), VEC3_T(0.0));

  Kokkos::View<Vec3 *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Atomic>> h_zone_gradient(&zone_gradient[0], zone_gradient.size());
  // create atomic mirror view for zone gradient
  //Kokkos::View<Vec3 *, Kokkos::SharedSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> 
	  auto d_zone_gradient = create_mirror_view ( space_t (), h_zone_gradient);
  Kokkos::deep_copy(d_zone_gradient, h_zone_gradient);

  Kokkos::parallel_for("gradzatz-2-cuda", num_local_corners, KOKKOS_LAMBDA (const int corner_idx) {
    if (d_corner_type[corner_idx] >= 1)
    { //  continue; // Only operate on interior corners
    int const zone_idx = d_c_to_z_map[corner_idx];
    int const point_idx = d_c_to_p_map[corner_idx];
    double const c_z_vol_ratio =
        d_corner_volume[corner_idx] / d_zone_volume[zone_idx];
	d_zone_gradient[zone_idx] += d_point_gradient[point_idx] * c_z_vol_ratio;
    }
   });

  Kokkos::fence();
  // copy zone and point gradient to host
  Kokkos::deep_copy(h_point_gradient, d_point_gradient);
  Kokkos::deep_copy(h_zone_gradient, d_zone_gradient);
#endif
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

  for (int point_idx = 0; point_idx < num_local_points; ++point_idx) {
    for (int const &corner_idx : p_to_c_map[point_idx]) {
      int const zone_idx = c_to_z_map[corner_idx];
      point_volume[point_idx] += corner_volume[corner_idx];
      point_gradient[point_idx] += csurf[corner_idx] * zone_field[zone_idx];
    }
  }

  mesh.points.gathscat(Ume::Comm::Op::SUM, point_volume);
  mesh.points.gathscat(Ume::Comm::Op::SUM, point_gradient);

  /*
    Divide by point control volume to get gradient.  If a point is on the outer
    perimeter of the mesh (POINT_TYPE=-1), subtract the outward normal component
    of the gradient using the point normals.
   */
  for (int point_idx = 0; point_idx < num_local_points; ++point_idx) {
    if (point_type[point_idx] > 0) {
      // Internal point
      point_gradient[point_idx] /= point_volume[point_idx];
    } else if (point_type[point_idx] == -1) {
      // Mesh boundary point
      double const ppdot =
          dotprod(point_gradient[point_idx], point_normal[point_idx]);
      point_gradient[point_idx] =
          (point_gradient[point_idx] - point_normal[point_idx] * ppdot) /
          point_volume[point_idx];
    }
  }
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
  for (int zone_idx = 0; zone_idx < num_local_zones; ++zone_idx) {
    if (zone_type[zone_idx] < 1)
      continue; // Only operate on local interior zones

    // Accumulate the (local) zone volume
    double zone_volume{0.0}; // Only need a local volume
    for (int const &corner_idx : z_to_c_map[zone_idx]) {
      zone_volume += corner_volume[corner_idx];
    }

    for (int const &corner_idx : z_to_c_map[zone_idx]) {
      int const point_idx = c_to_p_map[corner_idx];
      double const c_z_vol_ratio = corner_volume[corner_idx] / zone_volume;
      zone_gradient[zone_idx] += point_gradient[point_idx] * c_z_vol_ratio;
    }
  }

  mesh.zones.scatter(zone_gradient);
}

} // namespace Ume

