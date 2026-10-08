"""
Analytic (adjoint / suffix-pass) probability Jacobians for the map forward simulator.

For an expanded circuit with state preparation `rho`, layer operations `G_1, ..., G_n`
and POVM effect `E`, the outcome probability is

    p = E^T G_n ... G_1 rho.

Writing `rho_k = G_k ... G_1 rho` (forward states, `rho_0 = rho`) and
`e_k^T = E^T G_n ... G_{k+1}` (backward, or "suffix", effects, `e_n = E`), the
derivative with respect to a parameter `t` is

    dp/dt = sum_k e_k^T (dG_k/dt) rho_{k-1} + e_0^T (drho/dt) + rho_n^T (dE/dt).

Each term only needs the dense derivative of a single model member
(`ModelMember.deriv_wrt_params()`), so the cost of one Jacobian is a single forward
and a single backward sweep per circuit plus one small contraction per distinct
member, instead of one full re-simulation per model parameter.

This module currently supports models whose members can be represented as dense,
real superoperators (the `densitymx` family of evotypes) and whose members are all
stored directly in an :class:`ExplicitOpModel`.  Everything else falls back to the
finite-difference implementation in the evotype's `calclib`.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.tools import slicetools as _slct

#: Evotypes whose states/effects are real superkets and whose probabilities are linear in them.
SUPPORTED_EVOTYPES = ('densitymx_slow', 'densitymx')

#: Largest superoperator dimension (16 = two qubits) for which dense member derivatives and
#: per-element outer products are formed.  Beyond this they become too large to hold densely.
MAX_DENSE_DIM = 16


def supports_adjoint_dprobs(model, layout_atom):
    """
    Whether :func:`mapfill_dprobs_atom` can compute the Jacobian for `layout_atom` analytically.

    Parameters
    ----------
    model : Model
        The model being simulated.

    layout_atom : _MapCOPALayoutAtom
        The layout atom whose probability derivatives are requested.

    Returns
    -------
    bool
    """
    from pygsti.models.explicitmodel import ExplicitOpModel as _ExplicitOpModel
    if not isinstance(model, _ExplicitOpModel):
        return False  # implicit models build layer operators on the fly; not handled yet
    if model.evotype.name not in SUPPORTED_EVOTYPES or model.dim > MAX_DENSE_DIM:
        return False
    if len(model.instruments) > 0 or len(model.factories) > 0:
        return False  # instrument members and factory-built ops are not handled yet
    if not (all(lbl in model.preps for lbl in layout_atom.rho_labels)
            and all(lbl in model.operations for lbl in layout_atom.op_labels)
            and all(lbl in model.povms for lbl in layout_atom.povm_labels)):
        return False
    return True


def _member_derivative(member, wanted_cols):
    """
    Dense derivative of `member` restricted to the wanted model parameters.

    Returns `(cols, deriv)` where `cols` are column indices into the (restricted) Jacobian and
    `deriv` holds the matching columns of `member.deriv_wrt_params()`, or `None` if `member`
    depends on none of the wanted parameters.
    """
    gpindices = member.gpindices_as_array()
    if len(gpindices) == 0:
        return None
    cols = wanted_cols[gpindices]
    keep = cols >= 0
    if not _np.any(keep):
        return None
    deriv = _np.asarray(member.deriv_wrt_params(), dtype='d')
    return cols[keep], deriv[:, keep]


def mapfill_dprobs_atom(fwdsim, mx_to_fill, dest_indices, dest_param_indices, layout_atom, param_indices,
                        resource_alloc):
    """
    Fill `mx_to_fill` with analytic probability derivatives for the circuits of `layout_atom`.

    Arguments mirror those of `mapfill_dprobs_atom` in the `mapforwardsim_calc_*` modules
    (minus the finite-difference step size).  Callers should check
    :func:`supports_adjoint_dprobs` first.
    """
    model = fwdsim.model
    shared_mem_leader = resource_alloc.is_host_leader if (resource_alloc is not None) else True

    num_params = model.num_params
    if param_indices is None:
        param_indices = slice(0, num_params)
    param_indices = _slct.to_array(param_indices)
    if dest_param_indices is None:
        dest_param_indices = slice(0, len(param_indices))
    dest_param_indices = _slct.to_array(dest_param_indices)
    dest_indices = _slct.to_array(dest_indices)

    # wanted_cols[global param index] = column of the restricted Jacobian, or -1 if not wanted
    wanted_cols = -_np.ones(num_params, dtype=int)
    wanted_cols[param_indices] = _np.arange(len(param_indices))

    # Dense members and their parameter derivatives, computed once per atom.  Labels are mapped
    # to integers so the per-circuit loops below index arrays instead of hashing labels.
    preps = [model.preps[lbl] for lbl in layout_atom.rho_labels]
    ops = [model.operations[lbl] for lbl in layout_atom.op_labels]
    effects = [model._circuit_layer_operator(elbl, 'povm')  # same order as layout_atom.elabel_lookup
               for elbl in layout_atom.full_effect_labels]
    rho_index = {lbl: i for i, lbl in enumerate(layout_atom.rho_labels)}
    op_index = {lbl: i for i, lbl in enumerate(layout_atom.op_labels)}

    rho_vecs = [prep.to_dense('minimal').real for prep in preps]
    op_mxs = [op.to_dense('minimal').real for op in ops]
    op_mxs_T = [_np.ascontiguousarray(mx.T) for mx in op_mxs]
    effect_vecs = _np.array([effect.to_dense('minimal').real for effect in effects]).T  # (dim, nEffects)

    rho_derivs = [_member_derivative(prep, wanted_cols) for prep in preps]
    op_derivs = [_member_derivative(op, wanted_cols) for op in ops]
    effect_derivs = [_member_derivative(effect, wanted_cols) for effect in effects]

    dim = effect_vecs.shape[0]
    nEls = layout_atom.num_elements
    nOps = len(ops)
    eye_ops = _np.eye(nOps)
    jac = _np.zeros((nEls, len(param_indices)), 'd')

    # Per-element results of the sweeps, contracted with the member derivatives at the end:
    final_states = _np.zeros((nEls, dim), 'd')  # rho_n, for effect-parameter terms
    initial_effects = _np.zeros((nEls, dim), 'd')  # e_0, for state-prep-parameter terms
    rho_of_element = _np.zeros(nEls, dtype=int)
    effect_of_element = _np.zeros(nEls, dtype=int)
    # op_outer[o, el] = sum over occurrences k of op o of outer(e_k, rho_{k-1}), flattened
    op_outer = _np.zeros((nOps, nEls, dim * dim), 'd')

    forward_cache = {}  # iCache -> (rho index, layer indices, forward states) of a shared prefix
    for iDest, iStart, remainder, iCache in layout_atom.table.contents:
        # ---- forward pass: rho_0, ..., rho_n (reusing a cached prefix when available) ----
        if iStart is None:
            irho, remainder = rho_index[remainder[0]], remainder[1:]
            seq, states = [], [rho_vecs[irho]]
        else:
            irho, seq, states = forward_cache[iStart]
            seq, states = list(seq), list(states)
        for lbl in remainder:
            o = op_index[lbl]
            seq.append(o)
            states.append(op_mxs[o] @ states[-1])
        if iCache is not None:
            forward_cache[iCache] = (irho, seq, states)

        rows = layout_atom.elindices_by_expcircuit[iDest]
        eidx = layout_atom.elbl_indices_by_expcircuit[iDest]
        rho_of_element[rows] = irho
        effect_of_element[rows] = eidx
        final_states[rows] = states[-1]

        # ---- backward pass: e_n = E, e_{k-1} = G_k^T e_k (one column per outcome) ----
        e = effect_vecs[:, eidx]  # (dim, M)
        if seq:
            backward = [None] * len(seq)
            for k in range(len(seq) - 1, -1, -1):
                backward[k] = e
                e = op_mxs_T[seq[k]] @ e
            # sum_k outer(e_k, rho_{k-1}), grouped by which operation sits at layer k
            outers = _np.array(backward).transpose(0, 2, 1)[:, :, :, None] * _np.array(states[:-1])[:, None, None, :]
            grouped = eye_ops[:, seq] @ outers.reshape(len(seq), -1)  # (nOps, M * dim * dim)
            op_outer[:, rows, :] = grouped.reshape(nOps, len(rows), dim * dim)
        initial_effects[rows] = e.T

    # ---- contract with dense member derivatives ----
    for o, d in enumerate(op_derivs):
        if d is not None:
            cols, dG = d
            jac[:, cols] += op_outer[o] @ dG
    for i, d in enumerate(rho_derivs):
        if d is not None:
            cols, drho = d
            sel = rho_of_element == i
            jac[_np.ix_(sel, cols)] += initial_effects[sel] @ drho
    for j, d in enumerate(effect_derivs):
        if d is not None:
            cols, dE = d
            sel = effect_of_element == j
            jac[_np.ix_(sel, cols)] += final_states[sel] @ dE

    if shared_mem_leader:
        mx_to_fill[_np.ix_(dest_indices, dest_param_indices)] = jac
