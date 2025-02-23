# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities to help with mesh creation."""

from typing import List, Optional, Tuple
from absl import logging
import numpy as np

from tensorflow.dtensor import python as dtensor
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device


def _print_context(num_global_devices: int, num_clients: int, client_id: int,
                   device_type: str, mesh: dtensor.Mesh) -> None:
  logging.info('This is client %d of %d clients', client_id, num_clients)
  logging.info('Number of global %s devices: %d', device_type.upper(),
               num_global_devices)
  # pylint: disable=protected-access
  logging.info('Global device IDs: %s', mesh._global_device_ids)
  logging.info('Local device IDs: %s', mesh._local_device_ids)
  logging.info('Local devices: %s',
               [d.to_string() for d in mesh._local_devices])
  # pylint: enable=protected-access


def create_mesh(mesh_dims: Optional[List[Tuple[str, int]]] = None,
                mesh_name: str = '',
                devices: Optional[List[str]] = None,
                device_type: Optional[str] = None) -> dtensor.Mesh:
  """Creates a single-client mesh.

  If both `mesh_dims` and `devices` are specified, they must match each otehr.
  As a special case, when all arguments are missing, this creates a 1D CPU mesh
  with an empty name, assigning all available devices to that dimension.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples. Defaults to a single
      batch-parallel dimension called 'x' using all devices. As a special case,
      a single-element mesh_dims whose dim_size is -1 also uses all devices.
    mesh_name: Name of the created mesh. Defaults to ''.
    devices: String representations of devices to use. This is the device part
      of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available logical devices.
    device_type: If `devices` is missing, the type of devices to use. Defaults
      to 'CPU'.

  Returns:
    A single-client mesh created from specified or default arguments.
  """
  if devices is None:
    if device_type is None:
      device_type = 'CPU'
    devices = [
        tf_device.DeviceSpec.from_string(d.name)
        for d in tf_config.list_logical_devices(device_type)
    ]
  else:
    devices = [
        tf_device.DeviceSpec.from_string('/job:localhost/replica:0/task:0/' + d)
        for d in devices
    ]
    if device_type is None:
      device_type = devices[0].device_type
    if device_type.upper() != devices[0].device_type.upper():
      raise ValueError(
          f'Conflicting devices {str(devices)} and device_type {device_type}')
  if mesh_dims is None:
    mesh_dims = [('x', len(devices))]
  elif len(mesh_dims) == 1 and mesh_dims[0][1] == -1:
    # Replace -1 dim_size in a 1D mesh will the number of all devices.
    mesh_dims[0] = (mesh_dims[0][0], len(devices))

  dim_names = [d[0] for d in mesh_dims]
  shape = [d[1] for d in mesh_dims]
  global_device_ids = np.arange(len(devices)).reshape(shape)
  local_device_ids = np.ravel(global_device_ids).tolist()
  mesh = dtensor.Mesh(
      dim_names=dim_names,
      global_device_ids=global_device_ids,
      local_device_ids=local_device_ids,
      local_devices=devices,
      mesh_name=mesh_name)
  _print_context(
      num_global_devices=len(devices),
      num_clients=1,
      client_id=0,
      device_type=devices[0].device_type,
      mesh=mesh)
  return mesh


def create_distributed_mesh(mesh_dims: List[Tuple[str, int]],
                            mesh_name: str = '',
                            num_global_devices: Optional[int] = None,
                            num_clients: Optional[int] = None,
                            client_id: Optional[int] = None,
                            device_type: str = 'CPU') -> dtensor.Mesh:
  """Creates a single- or multi-client mesh.

  For CPU and GPU meshes, users can choose to use fewer local devices than what
  is available. If any argument is missing, it will be extracted from
  environment variables. The default values for these environment variables
  create a single-client mesh using all devices (common for unit tests).

  For TPU meshes, users should not specify any of the nullable arguments. The
  DTensor runtime will set these arguments automatically, using all TPU cores
  available in the entire cluster.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples.
    mesh_name: Name of the created mesh. Defaults to ''.
    num_global_devices: Number of devices in the DTensor cluster. Defaults to
      the corresponding environment variable.
    num_clients: Number of clients in the DTensor cluster. Defaults to the
      corresponding environment variable.
    client_id: This client's ID. Defaults to the corresponding environment
      variable.
    device_type: Type of device to build the mesh for. Defaults to 'CPU'.

  Returns:
    A single-client mesh created from specified or default arguments.
  """
  if device_type.upper() in ['CPU', 'GPU']:
    # For CPU and GPU meshes, user-specified args take precedence over env vars.
    # This is particularly useful on single clients when users want to create
    # meshes that use fewer logical devices than what's available.
    if num_global_devices is None:
      num_global_devices = dtensor.num_global_devices(device_type)
    if num_global_devices <= 0:
      raise ValueError(f'num_global_devices ({num_global_devices}) must be > 0')

    if num_clients is None:
      num_clients = dtensor.num_clients()
    if num_clients <= 0:
      raise ValueError(f'num_clients ({num_clients}) must be > 0')

    if client_id is None:
      client_id = dtensor.client_id()
    if client_id < 0:
      raise ValueError(f'client_id ({client_id}) must be >= 0')
    if client_id >= num_clients:
      raise ValueError(f'client_id ({client_id}) must be < {num_clients}')

    if num_global_devices % num_clients != 0:
      raise ValueError(f'num_global_devices ({num_global_devices}) must be '
                       f'divisible by num_clients ({num_clients})')
    num_local_devices = num_global_devices // num_clients

    # It's allowed to create a CPU or GPU mesh using fewer logical devices than
    # what's available. If so, just use the first N logical devices.
    num_available_devices = dtensor.num_local_devices(device_type)
    if num_local_devices > num_available_devices:
      raise ValueError(f'Not enough devices; {num_local_devices} needed, '
                       f'only {num_available_devices} available')
    local_devices = dtensor.local_devices(device_type,
                                          client_id)[:num_local_devices]

    dim_names = [d[0] for d in mesh_dims]
    shape = [d[1] for d in mesh_dims]
    global_device_ids = np.arange(num_global_devices).reshape(shape)
    flattened = np.ravel(global_device_ids).tolist()
    start_idx = num_local_devices * client_id
    local_device_ids = flattened[start_idx:start_idx + num_local_devices]

    mesh = dtensor.Mesh(
        dim_names=dim_names,
        global_device_ids=global_device_ids,
        local_device_ids=local_device_ids,
        local_devices=local_devices,
        mesh_name=mesh_name)
    _print_context(num_global_devices, num_clients, client_id, device_type,
                   mesh)
    return mesh

  if device_type.upper() == 'TPU':
    # TPU meshes can only be configured through environment variables that
    # reflect the actual TPU topology. Do not let users specify custom args.
    if num_global_devices is not None:
      raise ValueError(
          f'Do not specify num_global_devices for {device_type.upper()} meshes. '
          'It will be filled in automatically from environmental variables.'
          'See api.py for the list of environmental variables for DTensor.')
    if num_clients is not None:
      raise ValueError(
          f'Do not specify num_clients for {device_type.upper()} meshes. '
          'It will be filled in automatically from environmental variables.'
          'See api.py for the list of environmental variables for DTensor.')
    if client_id is not None:
      raise ValueError(
          f'Do not specify client_id for {device_type.upper()} meshes. '
          'It will be filled in automatically from environmental variables.'
          'See api.py for the list of environmental variables for DTensor.')
    dim_names = [mesh_dim[0] for mesh_dim in mesh_dims]
    shape = [mesh_dim[1] for mesh_dim in mesh_dims]
    mesh = tpu_util.create_tpu_mesh(dim_names, shape, mesh_name)
    _print_context(
        dtensor.num_global_devices(device_type), dtensor.num_clients(),
        dtensor.client_id(), device_type, mesh)
    return mesh

  raise ValueError(f'Device type {device_type} is not CPU, GPU or TPU')
