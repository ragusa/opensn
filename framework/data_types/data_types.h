// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

/**
 * \file
 * Enumerations and helper utilities describing data types that can be
 * stored inside a Varying object.
 */
namespace opensn
{

/// Enumeration of data-types supported by Varying
enum class VaryingDataType : int
{
  VOID = 0,            ///< Basically undefined or null
  ARBITRARY_BYTES = 1, ///< Basic sequence of bytes
  STRING = 2,          ///< Datatype mapping to std::string
  BOOL = 3,            ///< Datatype mapping to bool
  INTEGER = 4,         ///< Datatype mapping to int64_t
  FLOAT = 5,           ///< Datatype mapping to double
  USER_DATA = 6        ///< Datatype mapping to user data
};

/// Provides a string-name for an enumerated VaryingDataType.
std::string VaryingDataTypeStringName(VaryingDataType type);

class Varying;
class ByteArray;

} // namespace opensn
