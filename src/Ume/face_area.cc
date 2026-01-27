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
  \file Ume/face_area.cc
*/

#ifdef USE_CALI
#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#endif

#include "Ume/face_area.hh"
#include <Kokkos_Core.hpp>
namespace Ume {

using Mesh = SOA_Idx::Mesh;
using DBLV_T = DS_Types::DBLV_T;
using INTV_T = DS_Types::INTV_T;
using VEC3V_T = DS_Types::VEC3V_T;

void calc_face_area(Mesh &mesh, DBLV_T &face_area) {
  auto const &side_type = mesh.sides.mask;
  auto const &face_comm_type = mesh.faces.comm_type;
  auto const &s_to_f_map = mesh.ds->caccess_intv("m:s>f");
  auto const &s_to_s2_map = mesh.ds->caccess_intv("m:s>s2");
  auto const &surz = mesh.ds->caccess_vec3v("side_surz");

  int const sll = mesh.sides.size();
  int const sl = mesh.sides.local_size();

  std::fill(face_area.begin(), face_area.end(), 0.0);
  INTV_T side_tag(sll, 0);
 
 /* for (int s = 0; s < sl; ++s) {
    if (side_type[s] < 1)
      continue; // We want internal sides only
    if (side_tag[s] == 1)
      continue; // Already added this side via s2

    int const f = s_to_f_map[s];
    if (face_comm_type[f] < 3) { // Internal or master face
      double const side_area = vectormag(surz[s]); // Flat area
      face_area[f] += side_area;

      int const s2 = s_to_s2_map[s];
      side_tag[s2] = 1;
    }
  }

*/
#define HOST_SPACE Kokkos::HostSpace
using space_t = Kokkos::DefaultExecutionSpace::memory_space;

Kokkos::View<double *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  h_face_area(&face_area[0], face_area.size());
Kokkos::View<const int *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  h_s_to_f_map(&s_to_f_map[0], s_to_f_map.size());
Kokkos::View<const int *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  h_s_to_s2_map(&s_to_s2_map[0], s_to_s2_map.size());
Kokkos::View<const Vec3 *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  h_surz(&surz[0], surz.size());
Kokkos::View<int *, HOST_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>  h_side_tag(&side_tag[0], side_tag.size());
Kokkos::View<const short *, HOST_SPACE>  h_side_type(&side_type[0], side_type.size());
Kokkos::View<const int *, HOST_SPACE>  h_face_comm_type(&face_comm_type[0], face_comm_type.size());

auto d_face_area = create_mirror_view ( space_t () , h_face_area );
auto d_s_to_f_map = create_mirror_view ( space_t () , h_s_to_f_map );
auto d_s_to_s2_map = create_mirror_view ( space_t () , h_s_to_s2_map );
auto d_surz = create_mirror_view ( space_t () , h_surz );
auto d_side_tag = create_mirror_view ( space_t () , h_side_tag );
auto d_side_type = create_mirror_view ( space_t () , h_side_type );
auto d_face_comm_type = create_mirror_view ( space_t () , h_face_comm_type );

#ifdef KOKKOS_ENABLE_CUDA
Kokkos::deep_copy(d_face_area, h_face_area);
Kokkos::deep_copy(d_s_to_f_map, h_s_to_f_map);
Kokkos::deep_copy(d_s_to_s2_map, h_s_to_s2_map);
Kokkos::deep_copy(d_surz, h_surz);
Kokkos::deep_copy(d_side_tag, h_side_tag);
Kokkos::deep_copy(d_side_type, h_side_type);
Kokkos::deep_copy(d_face_comm_type, h_face_comm_type);
#endif
Kokkos::parallel_for("face_area", sl, KOKKOS_LAMBDA (const int s) {
    if (d_side_type[s] >= 1 && d_side_tag[s] != 1)
    {
    int const f = d_s_to_f_map[s];
    if (d_face_comm_type[f] < 3) { // Internal or master face
      double const side_area = vectormag(d_surz[s]); // Flat area
      d_face_area[f] += side_area;

      int const s2 = d_s_to_s2_map[s];
      d_side_tag[s2] = 1;
    }
    }
});

#ifdef KOKKOS_ENABLE_CUDA
Kokkos::fence();
Kokkos::deep_copy(h_face_area, d_face_area);
#endif
/*Kokkos::View<double *, Kokkos::HostSpace>  local_face_area(&face_area[0], face_area.size());
Kokkos::View<const int *, Kokkos::HostSpace>  local_s_to_f_map(&s_to_f_map[0], s_to_f_map.size());
Kokkos::View<const int *, Kokkos::HostSpace>  local_s_to_s2_map(&s_to_s2_map[0], s_to_s2_map.size());
Kokkos::View<const Vec3 *, Kokkos::HostSpace>  local_surz(&surz[0], surz.size());
Kokkos::View<int *, Kokkos::HostSpace>  local_side_tag(&side_tag[0], side_tag.size());
Kokkos::View<const short *, Kokkos::HostSpace>  local_side_type(&side_type[0], side_type.size());
Kokkos::View<const int *, Kokkos::HostSpace>  local_face_comm_type(&face_comm_type[0], face_comm_type.size());


Kokkos::parallel_for("face_area", sl, KOKKOS_LAMBDA (const int s) {
    if (local_side_type[s] >= 1 && local_side_tag[s] != 1)
    {
    int const f = local_s_to_f_map[s];
    if (local_face_comm_type[f] < 3) { // Internal or master face
      double const side_area = vectormag(local_surz[s]); // Flat area
      local_face_area[f] += side_area;

      int const s2 = local_s_to_s2_map[s];
      local_side_tag[s2] = 1;
    }
    }
});
*/
  mesh.faces.scatter(face_area);
}

} // namespace Ume
