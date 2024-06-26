// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "lbs_lua_utils.h"

#include "framework/runtime.h"
#include "framework/console/console.h"
#include "framework/logging/log.h"
#include "framework/lua.h"

#include "framework/mesh/logical_volume/logical_volume.h"
#include "lua/framework/math/functions/lua_spatial_material_function.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/lbs_solver.h"

using namespace opensn;

namespace opensnlua::lbs
{

RegisterLuaFunctionNamespace(AddPointSource, lbs, AddPointSource);
int
AddPointSource(lua_State* L)
{
  const std::string fname = "lbs.AddPointSource";
  LuaCheckArgs<int, int>(L, fname);

  // Process solver handle
  const auto solver_handle = LuaArg<int>(L, 1);
  auto& lbs_solver =
    opensn::GetStackItem<opensn::lbs::LBSSolver>(opensn::object_stack, solver_handle, __FUNCTION__);

  // Process point source handle
  auto pt_src_handle = LuaArg<int>(L, 2);
  lbs_solver.AddPointSource(std::move(opensn::GetStackItem<opensn::lbs::PointSource>(
    opensn::object_stack, pt_src_handle, __FUNCTION__)));
  return LuaReturn(L);
}

RegisterLuaFunctionNamespace(ClearPointSources, lbs, ClearPointSources);
int
ClearPointSources(lua_State* L)
{
  const std::string fname = "lbs.ClearPointSources";
  LuaCheckArgs<int>(L, fname);

  const auto solver_handle = LuaArg<int>(L, 1);
  auto& lbs_solver =
    opensn::GetStackItem<opensn::lbs::LBSSolver>(opensn::object_stack, solver_handle, __FUNCTION__);

  lbs_solver.ClearPointSources();
  opensn::log.Log() << "Cleared all point sources.";
  return LuaReturn(L);
}

RegisterLuaFunctionNamespace(AddDistributedSource, lbs, AddDistributedSource);
int
AddDistributedSource(lua_State* L)
{
  const std::string fname = "lbs.AddDistributedSource";
  LuaCheckArgs<int, int>(L, fname);

  const auto solver_handle = LuaArg<int>(L, 1);
  auto& lbs_solver =
    opensn::GetStackItem<opensn::lbs::LBSSolver>(object_stack, solver_handle, __FUNCTION__);

  const auto src_handle = LuaArg<int>(L, 2);
  lbs_solver.AddDistributedSource(std::move(
    opensn::GetStackItem<opensn::lbs::DistributedSource>(object_stack, src_handle, __FUNCTION__)));

  opensn::log.Log() << lbs_solver.TextName() << ": Added distributed source.";
  return LuaReturn(L);
}

RegisterLuaFunctionNamespace(ClearDistributedSources, lbs, ClearDistributedSources);
int
ClearDistributedSources(lua_State* L)
{
  const std::string fname = "lbs.ClearDistributedSources";
  LuaCheckArgs<int>(L, fname);

  // Process solver handle
  const auto solver_handle = LuaArg<int>(L, 1);
  auto& lbs_solver =
    opensn::GetStackItem<opensn::lbs::LBSSolver>(opensn::object_stack, solver_handle, __FUNCTION__);

  lbs_solver.ClearDistributedSources();
  opensn::log.Log() << "Cleared all distributed sources.";
  return LuaReturn(L);
}

} // namespace opensnlua::lbs
