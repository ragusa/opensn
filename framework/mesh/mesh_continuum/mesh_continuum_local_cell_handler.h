// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/cell/cell.h"

namespace opensn
{

/// Stores references to global cells to enable an iterator
class LocalCellHandler
{
public:
  std::vector<std::unique_ptr<Cell>>& native_cells_;

private:
  /// Constructor.
  explicit LocalCellHandler(std::vector<std::unique_ptr<Cell>>& native_cells)
    : native_cells_(native_cells)
  {
  }

public:
  static LocalCellHandler Create(std::vector<std::unique_ptr<Cell>>& native_cells)
  {
    return LocalCellHandler(native_cells);
  }

  /// Returns a reference to a local cell, given a local cell index.
  Cell& operator[](uint64_t cell_local_index);

  /// Returns a const reference to a local cell, given a local cell index.
  const Cell& operator[](uint64_t cell_local_index) const;

  /**
   * Get the total number of local cells.
   * @return Size of the native cell vector.
   */
  [[nodiscard]] size_t LocalCellCount() const { return native_cells_.size(); }

  /// Internal iterator class.
  class Iterator
  {
  private:
    LocalCellHandler& handler_;
    size_t index_;

  public:
    Iterator(LocalCellHandler& handler, size_t index) : handler_(handler), index_(index) {}

    Iterator& operator++()
    {
      ++index_;
      return *this;
    }

    Cell& operator*() { return *(handler_.native_cells_[index_]); }
    bool operator==(const Iterator& other) const { return index_ == other.index_; }
    bool operator!=(const Iterator& other) const { return index_ != other.index_; }
  };

  /// Internal const iterator class.
  class ConstIterator
  {
  private:
    const LocalCellHandler& handler_;
    size_t index_;

  public:
    ConstIterator(const LocalCellHandler& handler, size_t index) : handler_(handler), index_(index)
    {
    }

    ConstIterator operator++()
    {
      ++index_;
      return *this;
    }

    const Cell& operator*() { return *(handler_.native_cells_[index_]); }
    bool operator==(const ConstIterator& other) const { return index_ == other.index_; }
    bool operator!=(const ConstIterator& other) const { return index_ != other.index_; }
  };

  Iterator begin() { return {*this, 0}; }

  Iterator end() { return {*this, native_cells_.size()}; }

  [[nodiscard]] ConstIterator begin() const { return {*this, 0}; }

  [[nodiscard]] ConstIterator end() const { return {*this, native_cells_.size()}; }

  [[nodiscard]] size_t size() const { return native_cells_.size(); }
};

} // namespace opensn
