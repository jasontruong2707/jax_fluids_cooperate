from typing import Tuple, List

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.math.fft.helper_functions import create_slice, generate_indices, generate_permutations

Array = jax.Array

def parallel_fft(
        buffer: Array,
        split_factors: Tuple[int],
        split_axis_out: int = None
        ) -> Array:
    """Wrapper specifying a 
    fourier transform.

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param split_axis_out: _description_, defaults to None
    :type split_axis_out: int, optional
    :return: _description_
    :rtype: Array
    """
    velocity_hat = perform_fft(buffer, split_factors,
                               False, split_axis_out)

    return velocity_hat

def parallel_ifft(
        buffer_hat: Array,
        split_factors: Tuple[int],
        split_axis_out: int = None
        ) -> Array:
    """Wrapper specifying an inverse
    fourier transform.

    :param bufferr_hat: _description_
    :type bufferr_hat: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param split_axis_out: _description_, defaults to None
    :type split_axis_out: int, optional
    :return: _description_
    :rtype: Array
    """

    velocity = perform_fft(buffer_hat, split_factors,
                           True, split_axis_out)

    return jnp.real(velocity)

def perform_fft(
        buffer: Array,
        split_factors: Tuple[int],
        inverse: bool,
        split_axis_out: int = None
        ) -> Array:
    """Performs an (i)fft in parallel for a
    decomposed buffer. The domain
    must be decomposed along a single axis.
    If split_axis_out is provided, the returned
    buffer is decomposed along said axis.

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param inverse: _description_
    :type inverse: bool
    :param split_axis_out: _description_, defaults to None
    :type split_axis_out: int, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: Array
    """
    
    if inverse:
        fft_func = jnp.fft.ifftn
    else:
        fft_func = jnp.fft.fftn

    split_axis_in = np.argmax(np.array(split_factors))
    unsplit_axes = tuple([axis for axis in range(3) if axis != split_axis_in])
    for axis in unsplit_axes:
        if split_factors[axis] != 1:
            raise RuntimeError

    number_of_cells_device = buffer.shape[-3:]
    number_of_cells = tuple([int(number_of_cells_device[i]*split_factors[i]) for i in range(3)])

    no_subdomains = np.prod(np.array(split_factors))
    no_sends = no_subdomains - 1

    if split_axis_out != None:
        if split_axis_out == split_axis_in:
            split_axis_intermediate = np.roll(np.array([0,1,2]),-1)[split_axis_in]
        else:
            split_axis_intermediate = split_axis_out
    else:
        split_axis_out = split_axis_in
        split_axis_intermediate = np.roll(np.array([0,1,2]),-1)[split_axis_in]

    unsplit_axes = tuple([-3+axis for axis in unsplit_axes])
    buffer = fft_func(buffer, axes=unsplit_axes)

    permutations_list = generate_permutations(no_subdomains)

    cells_xi = number_of_cells[split_axis_intermediate]
    send_indices, reassemble_indices, keep_indices = generate_indices(cells_xi, no_subdomains)

    # SEND 
    device_id = jax.lax.axis_index(axis_name="i")
    receive_buffer_list = []
    for send_i in range(no_sends):
        indices = send_indices[send_i][device_id]
        s_ = create_slice(indices, split_axis_intermediate)
        send_buffer = buffer[s_]
        permutation = permutations_list[send_i]
        receive_buffer = jax.lax.ppermute(send_buffer, axis_name="i", perm=permutation)
        receive_buffer_list.append(receive_buffer)
    
    # REASSEMBLE
    indices = keep_indices[device_id]
    s_ = create_slice(indices, split_axis_intermediate)
    buffer = [buffer[s_]]
    for send_i in range(no_sends):
        buffer.append(receive_buffer_list[send_i])
    buffer = jnp.stack(buffer, axis=0)
    indices = reassemble_indices[device_id]
    buffer = jnp.concatenate([buffer[i] for i in indices], axis=-3+split_axis_in)
    
    buffer_out = fft_func(buffer, axes=(-3+split_axis_in,))

    if split_axis_in == split_axis_out:
        
        cells_xi = number_of_cells[split_axis_in]
        send_indices, reassemble_indices, keep_indices = generate_indices(cells_xi, no_subdomains)

        # SEND
        receive_buffer_list = []
        for send_i in range(no_sends):
            indices = send_indices[send_i][device_id]
            s_ = create_slice(indices, split_axis_in)
            send_buffer = buffer_out[s_]
            permutation = permutations_list[send_i]
            receive_buffer = jax.lax.ppermute(send_buffer, axis_name="i", perm=permutation)
            receive_buffer_list.append(receive_buffer)

        # REASSEMBLE
        indices = keep_indices[device_id]
        s_ = create_slice(indices, split_axis_in)
        buffer_out = [buffer_out[s_]]
        for send_i in range(no_sends):
            buffer_out.append(receive_buffer_list[send_i])
        buffer_out = jnp.stack(buffer_out, axis=0)
        indices = reassemble_indices[device_id]
        buffer_out = jnp.concatenate([buffer_out[i] for i in indices], axis=-3+split_axis_intermediate)

    return buffer_out
