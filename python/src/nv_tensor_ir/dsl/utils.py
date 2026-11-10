# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect

from .tracing import Location


def get_raw_func_locations(raw_func):
    unwrap_raw_func = inspect.unwrap(raw_func)
    try:
        src_file = inspect.getsourcefile(unwrap_raw_func)
        lines, start = inspect.getsourcelines(unwrap_raw_func)
    except (OSError, TypeError):
        return None, None, None
    if src_file is None:
        return None, None, None
    begin_loc = Location(filename=src_file, lineno=start)
    end_loc = Location(filename=src_file, lineno=start + len(lines) - 1)
    return src_file, begin_loc, end_loc
