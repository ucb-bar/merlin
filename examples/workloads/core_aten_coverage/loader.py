"""Mechanically selected Core ATen iteration cases; see the named generator below.

An optional adjacent merlin.iteration_workload_profile.v1 profile selects one
spec family with ``group``. Every tensor argument is an external input, including
weights; all results reach the single concatenated output.
"""

import hashlib
import itertools
import json
from pathlib import Path

import torch

# BEGIN GENERATED
# Generator: build_tools/scripts/make_core_aten_coverage.py
# bounded sha256: 8db0a20d05a10bc7591a4d024f66826ed2100c84ebb21ee176114d2b29eb74fb
# corpus_193 sha256: 02107dd9266eaa1109511484595ef12b4aebf7cf4b81cee707700220cae74d6b
SUITE_SHA256 = json.loads(
    '{"bounded":"8db0a20d05a10bc7591a4d024f66826ed2100c84ebb21ee176114d2b29eb74fb","corpus_193":"02107dd9266eaa1109511484595ef12b4aebf7cf4b81cee707700220cae74d6b"}'
)
ADMISSION_SHA256 = json.loads(
    '{"contract":"905fdf926d19de093f3ed3797ccd4a18bac2179054a77ac2838cbc9d5c81c629","facts":"242c2ff216417b7603b6b8493b0bb2e3169394803d7a83f8d489117500385dfb","host_capabilities":"4d968cee23e43b501f3dbef3c8d68b9822ff75f796daaed6296b139a07f879de","software_spec":"fd4574bd5893b5f4b7b3bce87ec4af41c8b17854296529b84c2d57a972d7dc43"}'
)
CENSUS = json.loads(
    '{"causal_decoder":{"loader_sha256":"6d73aacad53f1cf57a5a60fb27a54275f489bac494bef2648e69a054cbf07e0d","ops":["aten._softmax.default","aten.add.Scalar","aten.add.Tensor","aten.bmm.default","aten.cat.default","aten.clone.default","aten.div.Tensor","aten.embedding.default","aten.expand.default","aten.mean.dim","aten.mm.default","aten.mul.Tensor","aten.neg.default","aten.permute.default","aten.pow.Tensor_Scalar","aten.rsqrt.default","aten.sigmoid.default","aten.slice.Tensor","aten.split_with_sizes.default","aten.unsqueeze.default","aten.view.default"],"profile":{"ffn":80,"heads":6,"hidden":48,"kv_heads":2,"rope":true,"schema":"merlin.iteration_workload_profile.v1","seq":12,"workload_id":"causal_decoder"}},"causal_decoder_decode_step":{"loader_sha256":"6d73aacad53f1cf57a5a60fb27a54275f489bac494bef2648e69a054cbf07e0d","ops":["aten._softmax.default","aten.add.Scalar","aten.add.Tensor","aten.bmm.default","aten.cat.default","aten.clone.default","aten.div.Tensor","aten.embedding.default","aten.expand.default","aten.mean.dim","aten.mm.default","aten.mul.Tensor","aten.neg.default","aten.permute.default","aten.pow.Tensor_Scalar","aten.rsqrt.default","aten.sigmoid.default","aten.slice.Tensor","aten.split_with_sizes.default","aten.unsqueeze.default","aten.view.default"],"profile":{"cache_len":9,"decode_step":true,"ffn":80,"heads":6,"hidden":48,"kv_heads":2,"rope":true,"schema":"merlin.iteration_workload_profile.v1","seq":12,"workload_id":"causal_decoder"}},"coverage_mlp":{"loader_sha256":"a95a4c32f28f4c0e611df07b65de6310439ac04f53dc4f25a382b35770267fae","ops":["aten.addmm.default","aten.native_layer_norm.default","aten.permute.default","aten.relu.default"],"profile":null},"multimodal_policy":{"loader_sha256":"2db3951c68386a32451e69561e8e453b969f1ccbef9c2e5e77c06eb3f69d36b5","ops":["aten._softmax.default","aten.add.Tensor","aten.addmm.default","aten.bmm.default","aten.cat.default","aten.convolution.default","aten.cos.default","aten.div.Tensor","aten.embedding.default","aten.exp.default","aten.expand.default","aten.gelu.default","aten.mul.Tensor","aten.native_layer_norm.default","aten.permute.default","aten.sin.default","aten.unsqueeze.default","aten.view.default"],"profile":{"gelu_mlp":true,"queries":3,"schema":"merlin.iteration_workload_profile.v1","time_embedding":true,"tokens":13,"workload_id":"multimodal_policy"}},"residual_cnn":{"loader_sha256":"50669afe5feed509a9540363ce35d6cf0bba3c4ba3de13662f0a1a3e7faacff7","ops":["aten._native_batch_norm_legit_no_training.default","aten.add.Tensor","aten.addmm.default","aten.convolution.default","aten.max_pool2d_with_indices.default","aten.mean.dim","aten.permute.default","aten.relu.default"],"profile":{"batchnorm":true,"blocks":2,"bottleneck":true,"channels":12,"downsample":true,"schema":"merlin.iteration_workload_profile.v1","spatial_side":20,"stem":"conv7s2","stem_pool":true,"workload_id":"residual_cnn"}}}'
)
ROWS = [
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,3,3],"storage_offset":0,"stride":[18,9,3,1],"values":[[[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,5,5],"storage_offset":0,"stride":[25,25,5,1],"values":[[[[-0.75,-0.66796875,-0.58203125,-0.5,-0.416015625],[-0.333984375,-0.25,-0.1669921875,-0.08349609375,2.2351741790771484e-08],[0.08349609375,0.1669921875,0.25,0.333984375,0.416015625],[0.5,0.58203125,0.66796875,0.75,0.83203125],[0.91796875,1.0,1.0859375,1.1640625,1.25]]]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-0.75,-0.6328125,-0.515625],[-0.396484375,-0.279296875,-0.162109375],[-0.044189453125,0.07373046875,0.19140625]]],[[[0.30859375,0.42578125,0.54296875],[0.66015625,0.78125,0.8984375],[1.015625,1.1328125,1.25]]]]},[2],[1,1],[0,0],[1,1],false,[0,0],1,[true,true,true]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.convolution_backward.default::0c8f4903cf9fa791","dtype":"bfloat16","group":"contraction","overload":"aten.convolution_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,3,3],"storage_offset":1,"stride":[18,9,3,1],"values":[[[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0],[0.0,3.0,-2.0]],[[-1.0,0.0,3.0],[-2.0,-1.0,0.0],[3.0,-2.0,-1.0]]]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,5,5],"storage_offset":1,"stride":[25,25,5,1],"values":[[[[-2.0,-1.0,0.0,3.0,-2.0],[-1.0,0.0,3.0,-2.0,-1.0],[0.0,3.0,-2.0,-1.0,0.0],[3.0,-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0,-2.0]]]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,1,3,3],"storage_offset":1,"stride":[9,9,3,1],"values":[[[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0],[0.0,3.0,-2.0]]],[[[-1.0,0.0,3.0],[-2.0,-1.0,0.0],[3.0,-2.0,-1.0]]]]},[2],[1,1],[0,0],[1,1],false,[0,0],1,[true,true,true]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.convolution_backward.default::0dda80879fe7fd6f","dtype":"float16","group":"contraction","overload":"aten.convolution_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,3,3],"storage_offset":0,"stride":[18,9,3,1],"values":[[[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,5,5],"storage_offset":0,"stride":[25,25,5,1],"values":[[[[-0.75,-0.6666666865348816,-0.5833333134651184,-0.5,-0.4166666567325592],[-0.3333333134651184,-0.2499999850988388,-0.1666666567325592,-0.08333331346511841,2.2351741790771484e-08],[0.08333335816860199,0.1666666865348816,0.2499999701976776,0.3333333134651184,0.4166666269302368],[0.4999999701976776,0.5833333134651184,0.6666666269302368,0.75,0.8333333134651184],[0.9166666269302368,1.0,1.0833333730697632,1.1666666269302368,1.25]]]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-0.75,-0.6323529481887817,-0.5147058963775635],[-0.3970588147640228,-0.27941176295280457,-0.1617647111415863],[-0.04411764442920685,0.07352941483259201,0.19117647409439087]]],[[[0.30882352590560913,0.4264705777168274,0.5441176295280457],[0.6617646813392639,0.779411792755127,0.8970588445663452],[1.0147058963775635,1.1323529481887817,1.25]]]]},[2],[1,1],[0,0],[1,1],false,[0,0],1,[true,true,true]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.convolution_backward.default","dtype":"float32","group":"contraction","overload":"aten.convolution_backward.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[{"kind":"float","value":"-inf"},-1.0,0.0],[{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0]]},1],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten.cumsum.default::2dab27e62c66f835","dtype":"bfloat16","group":"contraction","overload":"aten.cumsum.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0.0,0.0,0.0],[0.0,0.0,0.0]]},1],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten.cumsum.default::1b2012496b004a72","dtype":"float16","group":"contraction","overload":"aten.cumsum.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.cumsum.default","dtype":"float32","group":"contraction","overload":"aten.cumsum.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0,0,0],[0,0,0]]},1],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int32"}}},"case_id":"aten.cumsum.default::6367e008810eb280","dtype":"int32","group":"contraction","overload":"aten.cumsum.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0,0,0],[0,0,1]]},0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int64"}}},"case_id":"aten.cumsum.default::74624fb3ccb606af","dtype":"int64","group":"contraction","overload":"aten.cumsum.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0,0,0],[0,0,1]]},1],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int8"}}},"case_id":"aten.cumsum.default::d137255a9df8d860","dtype":"int8","group":"contraction","overload":"aten.cumsum.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":{"kind":"float","value":"-inf"}},2.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar::4a0a13a5cdf89e83","dtype":"bfloat16","group":"elementwise_map","overload":"aten.div.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":false},2.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar::7da8e59aa96fc7ed","dtype":"bool","group":"elementwise_map","overload":"aten.div.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":{"kind":"float","value":"nan"}},1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar::1a47ff079862df69","dtype":"float16","group":"elementwise_map","overload":"aten.div.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},2.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar","dtype":"float32","group":"elementwise_map","overload":"aten.div.Scalar","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":1,"stride":[1],"values":[0]},-2.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar::8468ec10d8f8a580","dtype":"int32","group":"elementwise_map","overload":"aten.div.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},-2.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar::c2d8f4ca715d9c27","dtype":"int64","group":"elementwise_map","overload":"aten.div.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},-0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.div.Scalar::18f4aabdd90ecff3","dtype":"int8","group":"elementwise_map","overload":"aten.div.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":-0.75},0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar::1a3b09ab26193deb","dtype":"bfloat16","group":"elementwise_map","overload":"aten.mul.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":1,"stride":[1],"values":[false]},2.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar::94452874b9b1ff9a","dtype":"bool","group":"elementwise_map","overload":"aten.mul.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":{"kind":"float","value":"nan"}},1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar::07cae42be5e2cf57","dtype":"float16","group":"elementwise_map","overload":"aten.mul.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar","dtype":"float32","group":"elementwise_map","overload":"aten.mul.Scalar","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},0.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar::91c03f5fa1bdbeb7","dtype":"int32","group":"elementwise_map","overload":"aten.mul.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":0,"stride":[1],"values":[0]},0.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar::8eeae3cd627d991a","dtype":"int64","group":"elementwise_map","overload":"aten.mul.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},-0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mul.Scalar::745226b37f8badd6","dtype":"int8","group":"elementwise_map","overload":"aten.mul.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":{"kind":"float","value":"nan"}},-0.5],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Scalar::cbd0e6a172e58cf3","dtype":"bfloat16","group":"elementwise_map","overload":"aten.sub.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":1,"stride":[1],"values":[{"kind":"float","value":"nan"}]},0.0],"kind":"tuple"},"kwargs":{"alpha":0}},"case_id":"aten.sub.Scalar::3a9f26bdbf7440a3","dtype":"float16","group":"elementwise_map","overload":"aten.sub.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},0.5],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Scalar","dtype":"float32","group":"elementwise_map","overload":"aten.sub.Scalar","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},-2.0],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Scalar::9af7106e9cb12503","dtype":"int32","group":"elementwise_map","overload":"aten.sub.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},2.0],"kind":"tuple"},"kwargs":{"alpha":-1}},"case_id":"aten.sub.Scalar::5e6f51e8603970d6","dtype":"int64","group":"elementwise_map","overload":"aten.sub.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},1.0],"kind":"tuple"},"kwargs":{"alpha":0}},"case_id":"aten.sub.Scalar::5783ca329d63a1ac","dtype":"int8","group":"elementwise_map","overload":"aten.sub.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":0,"stride":[1],"values":[-0.75]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":0,"stride":[1],"values":[-0.75]}],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Tensor::43e496f3ee378deb","dtype":"bfloat16","group":"elementwise_map","overload":"aten.sub.Tensor","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":-0.75},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":-0.75}],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Tensor::5cad6aa69537d03c","dtype":"float16","group":"elementwise_map","overload":"aten.sub.Tensor","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0.050000011920928955,-0.3499999940395355,-0.75],[1.25,0.8500000238418579,0.44999998807907104]]}],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Tensor","dtype":"float32","group":"elementwise_map","overload":"aten.sub.Tensor","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":1,"stride":[1],"values":[0]},{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":1,"stride":[1],"values":[0]}],"kind":"tuple"},"kwargs":{"alpha":0}},"case_id":"aten.sub.Tensor::03c9f285d4fbdd15","dtype":"int32","group":"elementwise_map","overload":"aten.sub.Tensor","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0},{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":0}],"kind":"tuple"},"kwargs":{"alpha":0}},"case_id":"aten.sub.Tensor::0b84e82b25b9e208","dtype":"int64","group":"elementwise_map","overload":"aten.sub.Tensor","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":0,"stride":[1],"values":[0]},{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":0,"stride":[1],"values":[0]}],"kind":"tuple"},"kwargs":{"alpha":2}},"case_id":"aten.sub.Tensor::3a7908b0ecbcd881","dtype":"int8","group":"elementwise_map","overload":"aten.sub.Tensor","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten._to_copy.default::06ee6703581eb770","dtype":"bfloat16","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[false,false,false],[false,false,false]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bool"},"non_blocking":true}},"case_id":"aten._to_copy.default::2d79a944de128d65","dtype":"bool","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0}]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"complex64"}}},"case_id":"aten._to_copy.default::2fd605118c9f4dc9","dtype":"complex64","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0.5,0.5,0.5],[0.5,0.5,0.5]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten._to_copy.default::2eed9591d7389cd4","dtype":"float16","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float64"}}},"case_id":"aten._to_copy.default","dtype":"float32","group":"movement","overload":"aten._to_copy.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int32"},"non_blocking":true}},"case_id":"aten._to_copy.default::5a1a48364160f9f4","dtype":"int32","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[-2,-1,0],[3,-2,-1]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int64"}}},"case_id":"aten._to_copy.default::0cef581513522804","dtype":"int64","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int8"},"non_blocking":true}},"case_id":"aten._to_copy.default::08dd12372d4128c2","dtype":"int8","group":"movement","overload":"aten._to_copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::2766df9e3298fb82","dtype":"bfloat16","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[true,true,true],[true,true,true]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::30b4bed99551ac69","dtype":"bool","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":1.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0}]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::25fe9e8ae01dfa8d","dtype":"complex64","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-65504.0,-1.0,0.0],[1.0,65504.0,-65504.0]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::39e24fc299cd6164","dtype":"float16","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default","dtype":"float32","group":"movement","overload":"aten.alias.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-2147483648,-1,0],[1,2147483647,-2147483648]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::0d5593b893a66dc9","dtype":"int32","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0,0,0],[0,0,0]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::05b606a5425b693c","dtype":"int64","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0,0,0],[0,0,0]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.alias.default::890e1093d9f23ba9","dtype":"int8","group":"movement","overload":"aten.alias.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":1,"stride":[4,1],"values":[[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::49bb809bcc196071","dtype":"bfloat16","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[false,false,false,false],[false,false,false,false],[false,false,false,false]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::384a6ee409470373","dtype":"bool","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":1,"stride":[4,1],"values":[[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::0049d5efdeb7abef","dtype":"complex64","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[-65504.0,-1.0,0.0,1.0],[65504.0,-65504.0,-1.0,0.0],[1.0,65504.0,-65504.0,-1.0]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::22434b13d8c9c842","dtype":"float16","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[-0.75,-0.5681818127632141,-0.3863636255264282,-0.20454543828964233],[-0.022727251052856445,0.15909093618392944,0.34090906381607056,0.5227272510528564],[0.7045454382896423,0.8863636255264282,1.0681817531585693,1.25]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default","dtype":"float32","group":"movement","overload":"aten.as_strided.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[-2,-1,0,3],[-2,-1,0,3],[-2,-1,0,3]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::26186d4b2088392a","dtype":"int32","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[0,0,0,0],[0,0,0,0],[0,0,1,1]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::2b944c9ab5e22d6b","dtype":"int64","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[-128,-1,0,1],[127,-128,-1,0],[1,127,-128,-1]]},[2,2],[4,1],1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.as_strided.default::2db301929fcfc778","dtype":"int8","group":"movement","overload":"aten.as_strided.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":1,"stride":[6,3,1],"values":[[[-0.0,0.0,-0.0],[0.0,-0.0,0.0]]]},[1,1,1,1],0.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::3fabe83a352126c0","dtype":"bfloat16","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":1,"stride":[6,3,1],"values":[[[false,false,false],[false,false,false]]]},[1,1,1,1],-1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::1c44b30c61de54ff","dtype":"bool","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":1,"stride":[6,3,1],"values":[[[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}]]]},[1,1,1,1],1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::04bb43ad73732b83","dtype":"complex64","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":0,"stride":[6,0,1],"values":[[[-0.75,-0.35009765625,0.04998779296875],[-0.75,-0.35009765625,0.04998779296875]]]},[1,1,1,1],0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::04b19c45b74e93cc","dtype":"float16","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":0,"stride":[6,3,1],"values":[[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]]},[1,1,1,1],0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default","dtype":"float32","group":"movement","overload":"aten.constant_pad_nd.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":0,"stride":[6,3,1],"values":[[[0,0,0],[0,0,0]]]},[1,1,1,1],0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::90b5110a480f780d","dtype":"int32","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":0,"stride":[6,0,1],"values":[[[0,0,0],[0,0,0]]]},[1,1,1,1],0.5],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::0011e7384b5b4521","dtype":"int64","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,2,3],"storage_offset":1,"stride":[6,3,1],"values":[[[-128,-1,0],[1,127,-128]]]},[1,1,1,1],-1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.constant_pad_nd.default::09cff5f3be720e71","dtype":"int8","group":"movement","overload":"aten.constant_pad_nd.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0.0,0.0,0.0],[0.0,0.0,0.0]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0.0,0.0,0.0],[0.0,0.0,0.0]]},true],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::04e9af5b5fdb59f7","dtype":"bfloat16","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[false,true,false],[false,true,false]]},{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[false,true,false],[false,true,false]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::4fda6b7caf8a2252","dtype":"bool","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0}]]},{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0}]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::130885505d678b7f","dtype":"complex64","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0.5,0.5,0.5],[0.5,0.5,0.5]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0.5,0.5,0.5],[0.5,0.5,0.5]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::0fa5c950006fa4fd","dtype":"float16","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0.050000011920928955,-0.3499999940395355,-0.75],[1.25,0.8500000238418579,0.44999998807907104]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default","dtype":"float32","group":"movement","overload":"aten.copy.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0,0,0],[0,0,1]]},{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0,0,0],[1,0,0]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::7d98ba8482ac34ec","dtype":"int32","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0,0,0],[0,0,0]]},{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0,0,0],[0,0,0]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::3530de886d0a1932","dtype":"int64","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0,0,0],[0,0,0]]},{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[0,0,0],[0,0,0]]},false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.copy.default::323b79e168a7dc20","dtype":"int8","group":"movement","overload":"aten.copy.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[8,2],"values":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::10fa1d51e9a68b77","dtype":"bfloat16","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":1,"stride":[4,1],"values":[[false,true,false,true],[false,true,false,true],[false,true,false,true]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::43172ef03be7a641","dtype":"bool","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":1,"stride":[4,1],"values":[[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}],[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}],[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::07b132ba5c670915","dtype":"complex64","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[-65504.0,-1.0,0.0,1.0],[65504.0,-65504.0,-1.0,0.0],[1.0,65504.0,-65504.0,-1.0]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::3d58cab0ddd5a228","dtype":"float16","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[4,1],"values":[[-0.75,-0.5681818127632141,-0.3863636255264282,-0.20454543828964233],[-0.022727251052856445,0.15909093618392944,0.34090906381607056,0.5227272510528564],[0.7045454382896423,0.8863636255264282,1.0681817531585693,1.25]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default","dtype":"float32","group":"movement","overload":"aten.diagonal.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[0,1],"values":[[0,0,0,0],[0,0,0,0],[0,0,0,0]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::213ece470b988e5f","dtype":"int32","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":0,"stride":[0,1],"values":[[0,0,0,0],[0,0,0,0],[0,0,0,0]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::21e4573d94a8e4ab","dtype":"int64","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[3,4],"storage_offset":1,"stride":[4,1],"values":[[0,0,0,0],[0,0,0,0],[0,0,0,0]]},0,0,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.diagonal.default::4a646fdea444ae05","dtype":"int8","group":"movement","overload":"aten.diagonal.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[{"kind":"float","value":"-inf"},-1.0,0.0],[{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0]]},1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::044a355bdbcb2ac3","dtype":"bfloat16","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[false,false,false],[false,false,false]]},0.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::313adc8c15ab1b0f","dtype":"bool","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":-3.4028234663852886e+38},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":1.0},{"imag":0.0,"kind":"complex","real":3.4028234663852886e+38},{"imag":0.0,"kind":"complex","real":-3.4028234663852886e+38}]]},3.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::2df1dfeb7d256940","dtype":"complex64","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0.5,0.5,0.5],[0.5,0.5,0.5]]},0.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::1bef6a7a602e0242","dtype":"float16","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},3.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar","dtype":"float32","group":"movement","overload":"aten.fill.Scalar","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[0,0,0],[0,0,0]]},1.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::04287aec1f00cbf2","dtype":"int32","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808]]},0.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::10b0d127c0a1f1a8","dtype":"int64","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]},3.0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.fill.Scalar::078745846b05b68e","dtype":"int8","group":"movement","overload":"aten.fill.Scalar","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[{"kind":"float","value":"nan"},-1.0,0.0],[1.0,{"kind":"float","value":"nan"},-1.0]]},[0]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::0a51fa712ab6b786","dtype":"bfloat16","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[false,true,false],[true,false,true]]},[]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::038add747f3a420c","dtype":"bool","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":1.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0}]]},[1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::08f42e78623eb2e1","dtype":"complex64","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-65504.0,-1.0,0.0],[1.0,65504.0,-65504.0]]},[1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::235b2bdc7fcce903","dtype":"float16","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},[1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default","dtype":"float32","group":"movement","overload":"aten.flip.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[-2,-1,0],[3,-2,-1]]},[0,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::461870e8c0c783c5","dtype":"int32","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0,0,0],[0,0,1]]},[]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::0bd97b866f8b698a","dtype":"int64","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]},[-1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.flip.default::0f91ba01a4c52714","dtype":"int8","group":"movement","overload":"aten.flip.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[],0.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten.full.default::3a26a30b04f3a3b0","dtype":"bfloat16","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[5],1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bool"}}},"case_id":"aten.full.default::069594f69dc86dc7","dtype":"bool","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[2,3],1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"complex64"}}},"case_id":"aten.full.default::4c9a4d5634d5bdd2","dtype":"complex64","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[2,3],0.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten.full.default::01866a0f93fe3100","dtype":"float16","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[2,3],1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float32"}}},"case_id":"aten.full.default","dtype":"float32","group":"movement","overload":"aten.full.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[1],1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int32"}}},"case_id":"aten.full.default::0f56daf8196ab245","dtype":"int32","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[2,1,5],0.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int64"}}},"case_id":"aten.full.default::5c63d642f37e16e8","dtype":"int64","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[[],1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int8"}}},"case_id":"aten.full.default::01647221d339751d","dtype":"int8","group":"movement","overload":"aten.full.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]},1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten.full_like.default::269da93623f63cc3","dtype":"bfloat16","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[true,true,true],[true,true,true]]},1.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bool"}}},"case_id":"aten.full_like.default::017a4ab4e308d0c5","dtype":"bool","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}]]},1.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"complex64"}}},"case_id":"aten.full_like.default::0ec87e6a20de73b6","dtype":"complex64","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.0,0.0,-0.0],[0.0,-0.0,0.0]]},1.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten.full_like.default::2da516bf326a82ae","dtype":"float16","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float64"}}},"case_id":"aten.full_like.default","dtype":"float32","group":"movement","overload":"aten.full_like.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0,0,0],[0,0,1]]},1.25],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int32"}}},"case_id":"aten.full_like.default::18488f2e9c297004","dtype":"int32","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0,0,0],[0,0,1]]},-1.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int64"}}},"case_id":"aten.full_like.default::118bfc117d59c547","dtype":"int64","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[-128,-1,0],[1,127,-128]]},-1.0],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int8"}}},"case_id":"aten.full_like.default::062082f03d6b762d","dtype":"int8","group":"movement","overload":"aten.full_like.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":0,"stride":[4,4,0],"values":[[[{"kind":"float","value":"nan"},{"kind":"float","value":"nan"},{"kind":"float","value":"nan"},{"kind":"float","value":"nan"}]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default::0e713810b1f36512","dtype":"bfloat16","group":"movement","overload":"aten.reflection_pad1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":0,"stride":[4,4,1],"values":[[[{"imag":0.0,"kind":"complex","real":-0.75},{"imag":0.0,"kind":"complex","real":-0.08333331346511841},{"imag":0.0,"kind":"complex","real":0.5833333134651184},{"imag":0.0,"kind":"complex","real":1.25}]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default::1af4519804a6a637","dtype":"complex64","group":"movement","overload":"aten.reflection_pad1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":1,"stride":[4,4,1],"values":[[[0.0,0.0,0.0,0.0]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default::1c1ceb4f684d1197","dtype":"float16","group":"movement","overload":"aten.reflection_pad1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":0,"stride":[4,4,1],"values":[[[-0.75,-0.08333331346511841,0.5833333134651184,1.25]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default","dtype":"float32","group":"movement","overload":"aten.reflection_pad1d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":1,"stride":[4,4,1],"values":[[[0,0,0,0]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default::478b50e16492a3f7","dtype":"int32","group":"movement","overload":"aten.reflection_pad1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":0,"stride":[4,4,0],"values":[[[0,0,0,0]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default::52c862ca28fb699e","dtype":"int64","group":"movement","overload":"aten.reflection_pad1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,1,4],"storage_offset":0,"stride":[4,4,0],"values":[[[0,0,0,0]]]},[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad1d.default::20b1d6759e087bab","dtype":"int8","group":"movement","overload":"aten.reflection_pad1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,0,1],"values":[[[[{"kind":"float","value":"-inf"},-1.0,0.0],[{"kind":"float","value":"-inf"},-1.0,0.0],[{"kind":"float","value":"-inf"},-1.0,0.0]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default::0956a93ed9e23c80","dtype":"bfloat16","group":"movement","overload":"aten.reflection_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[18,18,6,2],"values":[[[[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":0.0}]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default::077b27e92fc3ece8","dtype":"complex64","group":"movement","overload":"aten.reflection_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[18,18,6,2],"values":[[[[{"kind":"float","value":"-inf"},-1.0,0.0],[{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0],[0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"}]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default::19945ffd8eb09304","dtype":"float16","group":"movement","overload":"aten.reflection_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-0.75,-0.5,-0.25],[0.0,0.25,0.5],[0.75,1.0,1.25]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default","dtype":"float32","group":"movement","overload":"aten.reflection_pad2d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-2147483648,-1,0],[1,2147483647,-2147483648],[-1,0,1]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default::08abfd52fa2ca2e6","dtype":"int32","group":"movement","overload":"aten.reflection_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,0,1],"values":[[[[0,0,0],[0,0,0],[0,0,0]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default::2137344bc3702822","dtype":"int64","group":"movement","overload":"aten.reflection_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-128,-1,0],[1,127,-128],[-1,0,1]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad2d.default::3ab07828502539a8","dtype":"int8","group":"movement","overload":"aten.reflection_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,0,3,1],"values":[[[[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38],[-1.0,0.0,1.0]],[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38],[-1.0,0.0,1.0]],[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38],[-1.0,0.0,1.0]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default::37aa882b446f677c","dtype":"bfloat16","group":"movement","overload":"aten.reflection_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,9,3,1],"values":[[[[[{"imag":0.0,"kind":"complex","real":-0.75},{"imag":0.0,"kind":"complex","real":-0.6730769276618958},{"imag":0.0,"kind":"complex","real":-0.5961538553237915}],[{"imag":0.0,"kind":"complex","real":-0.5192307829856873},{"imag":0.0,"kind":"complex","real":-0.4423076808452606},{"imag":0.0,"kind":"complex","real":-0.36538460850715637}],[{"imag":0.0,"kind":"complex","real":-0.28846150636672974},{"imag":0.0,"kind":"complex","real":-0.2115384340286255},{"imag":0.0,"kind":"complex","real":-0.13461536169052124}]],[[{"imag":0.0,"kind":"complex","real":-0.057692281901836395},{"imag":0.0,"kind":"complex","real":0.01923079788684845},{"imag":0.0,"kind":"complex","real":0.0961538776755333}],[{"imag":0.0,"kind":"complex","real":0.17307695746421814},{"imag":0.0,"kind":"complex","real":0.2499999701976776},{"imag":0.0,"kind":"complex","real":0.32692304253578186}],[{"imag":0.0,"kind":"complex","real":0.4038461148738861},{"imag":0.0,"kind":"complex","real":0.48076921701431274},{"imag":0.0,"kind":"complex","real":0.557692289352417}]],[[{"imag":0.0,"kind":"complex","real":0.6346153616905212},{"imag":0.0,"kind":"complex","real":0.7115384340286255},{"imag":0.0,"kind":"complex","real":0.7884615063667297}],[{"imag":0.0,"kind":"complex","real":0.865384578704834},{"imag":0.0,"kind":"complex","real":0.942307710647583},{"imag":0.0,"kind":"complex","real":1.0192307233810425}],[{"imag":0.0,"kind":"complex","real":1.0961538553237915},{"imag":0.0,"kind":"complex","real":1.173076868057251},{"imag":0.0,"kind":"complex","real":1.25}]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default::006c2ada709d5506","dtype":"complex64","group":"movement","overload":"aten.reflection_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[54,54,18,6,2],"values":[[[[[-0.0,0.0,-0.0],[0.0,-0.0,0.0],[-0.0,0.0,-0.0]],[[0.0,-0.0,0.0],[-0.0,0.0,-0.0],[0.0,-0.0,0.0]],[[-0.0,0.0,-0.0],[0.0,-0.0,0.0],[-0.0,0.0,-0.0]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default::23347d0c0af389e2","dtype":"float16","group":"movement","overload":"aten.reflection_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,9,3,1],"values":[[[[[-0.75,-0.6730769276618958,-0.5961538553237915],[-0.5192307829856873,-0.4423076808452606,-0.36538460850715637],[-0.28846150636672974,-0.2115384340286255,-0.13461536169052124]],[[-0.057692281901836395,0.01923079788684845,0.0961538776755333],[0.17307695746421814,0.2499999701976776,0.32692304253578186],[0.4038461148738861,0.48076921701431274,0.557692289352417]],[[0.6346153616905212,0.7115384340286255,0.7884615063667297],[0.865384578704834,0.942307710647583,1.0192307233810425],[1.0961538553237915,1.173076868057251,1.25]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default","dtype":"float32","group":"movement","overload":"aten.reflection_pad3d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,0,3,1],"values":[[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default::17d31861db8cadc5","dtype":"int32","group":"movement","overload":"aten.reflection_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,9,3,1],"values":[[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,1],[1,1,1]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default::18bca99bb37b3c3f","dtype":"int64","group":"movement","overload":"aten.reflection_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,9,3,1],"values":[[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,1],[1,1,1]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.reflection_pad3d.default::00a4bc33d1262294","dtype":"int8","group":"movement","overload":"aten.reflection_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::0cb8f2552283df1d","dtype":"bfloat16","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[false,true,false],[true,false,true]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::057b00749cbd0743","dtype":"bool","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":1,"stride":[3,1],"values":[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":1.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0}]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::065a0d1ae7daef28","dtype":"complex64","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[0.5,0.5,0.5],[0.5,0.5,0.5]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::0f632745cbb53ac0","dtype":"float16","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default","dtype":"float32","group":"movement","overload":"aten.repeat.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::2ccfd9a3e6029489","dtype":"int32","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::17d7c937aa9fe346","dtype":"int64","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]},[2,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.repeat.default::31eaaf14d9fcb30e","dtype":"int8","group":"movement","overload":"aten.repeat.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":1,"stride":[9,9,3,1],"values":[[[[-0.0,0.0,-0.0],[0.0,-0.0,0.0],[-0.0,0.0,-0.0]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default::172ef95496c73744","dtype":"bfloat16","group":"movement","overload":"aten.replication_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":1,"stride":[9,9,3,1],"values":[[[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}}]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default::1478b4837720d268","dtype":"complex64","group":"movement","overload":"aten.replication_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-65504.0,-1.0,0.0],[1.0,65504.0,-65504.0],[-1.0,0.0,1.0]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default::3a1fcb71a1d82aa0","dtype":"float16","group":"movement","overload":"aten.replication_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-0.75,-0.5,-0.25],[0.0,0.25,0.5],[0.75,1.0,1.25]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default","dtype":"float32","group":"movement","overload":"aten.replication_pad2d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,0,1],"values":[[[[0,0,0],[0,0,0],[0,0,0]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default::4b3bef5a4f10b0ba","dtype":"int32","group":"movement","overload":"aten.replication_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":1,"stride":[9,9,3,1],"values":[[[[-2,-1,0],[3,-2,-1],[0,3,-2]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default::1b05a93ed73aa269","dtype":"int64","group":"movement","overload":"aten.replication_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":1,"stride":[9,9,3,1],"values":[[[[-2,-1,0],[3,-2,-1],[0,3,-2]]]]},[1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad2d.default::27be0727cc9931b5","dtype":"int8","group":"movement","overload":"aten.replication_pad2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,0,3,1],"values":[[[[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0],[0.0,3.0,-2.0]],[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0],[0.0,3.0,-2.0]],[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0],[0.0,3.0,-2.0]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default::0671d3697f1b14dc","dtype":"bfloat16","group":"movement","overload":"aten.replication_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":1,"stride":[27,27,9,3,1],"values":[[[[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}}]],[[{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0}]],[[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}}],[{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"inf"}}],[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"-inf"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default::4c6b10799c0549e1","dtype":"complex64","group":"movement","overload":"aten.replication_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":1,"stride":[27,27,9,3,1],"values":[[[[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default::058eafd1667a2d07","dtype":"float16","group":"movement","overload":"aten.replication_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,9,3,1],"values":[[[[[-0.75,-0.6730769276618958,-0.5961538553237915],[-0.5192307829856873,-0.4423076808452606,-0.36538460850715637],[-0.28846150636672974,-0.2115384340286255,-0.13461536169052124]],[[-0.057692281901836395,0.01923079788684845,0.0961538776755333],[0.17307695746421814,0.2499999701976776,0.32692304253578186],[0.4038461148738861,0.48076921701431274,0.557692289352417]],[[0.6346153616905212,0.7115384340286255,0.7884615063667297],[0.865384578704834,0.942307710647583,1.0192307233810425],[1.0961538553237915,1.173076868057251,1.25]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default","dtype":"float32","group":"movement","overload":"aten.replication_pad3d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,0,3,1],"values":[[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default::0c2dab25c1237c88","dtype":"int32","group":"movement","overload":"aten.replication_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,0,3,1],"values":[[[[[-2,-1,0],[3,-2,-1],[0,3,-2]],[[-2,-1,0],[3,-2,-1],[0,3,-2]],[[-2,-1,0],[3,-2,-1],[0,3,-2]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default::12b0b5e42e50c6c6","dtype":"int64","group":"movement","overload":"aten.replication_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,1,3,3,3],"storage_offset":0,"stride":[27,27,9,3,1],"values":[[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,1],[1,1,1]]]]]},[1,1,1,1,1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.replication_pad3d.default::2c9aea680019aa55","dtype":"int8","group":"movement","overload":"aten.replication_pad3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten.scalar_tensor.default::7d5c8bb8c69d0f7c","dtype":"bfloat16","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bool"}}},"case_id":"aten.scalar_tensor.default::4d15133a08a188a3","dtype":"bool","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"complex64"}}},"case_id":"aten.scalar_tensor.default::89541b3285fd74d7","dtype":"complex64","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten.scalar_tensor.default::e9b1b4a69b35cb1a","dtype":"float16","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float32"}}},"case_id":"aten.scalar_tensor.default","dtype":"float32","group":"movement","overload":"aten.scalar_tensor.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int32"}}},"case_id":"aten.scalar_tensor.default::172ff71c3469986e","dtype":"int32","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int64"}}},"case_id":"aten.scalar_tensor.default::2d57b9c35a5cec30","dtype":"int64","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[2.5],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int8"}}},"case_id":"aten.scalar_tensor.default::2984fd68c5f628b8","dtype":"int8","group":"movement","overload":"aten.scalar_tensor.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[24,8,2],"values":[[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]],[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]]]},2,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::03c707e8e370d0cd","dtype":"bfloat16","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[0,4,1],"values":[[[false,true,false,true],[false,true,false,true],[false,true,false,true]],[[false,true,false,true],[false,true,false,true],[false,true,false,true]]]},2,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::046ff8670056782c","dtype":"bool","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[12,4,1],"values":[[[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}],[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}],[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}]],[[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}],[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}],[{"imag":3.0,"kind":"complex","real":-2.0},{"imag":0.5,"kind":"complex","real":-0.0},{"imag":-0.0,"kind":"complex","real":0.5},{"imag":-2.0,"kind":"complex","real":3.0}]]]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::002d8fcbda080cc2","dtype":"complex64","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[24,8,2],"values":[[[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0]],[[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0]]]},-3,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::23ea2871f2987adf","dtype":"float16","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[12,4,1],"values":[[[-0.75,-0.6630434989929199,-0.5760869383811951,-0.489130437374115],[-0.4021739065647125,-0.31521737575531006,-0.2282608598470688,-0.14130434393882751],[-0.05434781312942505,0.03260871022939682,0.11956523358821869,0.20652174949645996]],[[0.29347825050354004,0.3804347515106201,0.4673912823200226,0.554347813129425],[0.6413043141365051,0.72826087474823,0.8152173757553101,0.9021738767623901],[0.989130437374115,1.0760869979858398,1.16304349899292,1.25]]]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int","dtype":"float32","group":"movement","overload":"aten.select.int","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[0,4,1],"values":[[[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0]]]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::0ad843a4bb6fe202","dtype":"int32","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[12,4,1],"values":[[[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,1,1,1]]]},-3,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::50068c9d99e2d015","dtype":"int64","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3,4],"storage_offset":0,"stride":[24,8,2],"values":[[[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0]]]},2,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select.int::2cc19756818c5153","dtype":"int8","group":"movement","overload":"aten.select.int","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[1],"values":[-2.0,-1.0]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::0ad8737e383eeea4","dtype":"bfloat16","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[true,true,true],[true,true,true]]},{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[1],"values":[true,true]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::58719765dadcbf95","dtype":"bool","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[6,2],"values":[[{"imag":0.0,"kind":"complex","real":-3.4028234663852886e+38},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":1.0},{"imag":0.0,"kind":"complex","real":3.4028234663852886e+38},{"imag":0.0,"kind":"complex","real":-3.4028234663852886e+38}]]},{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[2],"values":[{"imag":0.0,"kind":"complex","real":-3.4028234663852886e+38},{"imag":0.0,"kind":"complex","real":-1.0}]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::5e74e1f0a29db7fb","dtype":"complex64","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[{"kind":"float","value":"nan"},-1.0,0.0],[1.0,{"kind":"float","value":"nan"},-1.0]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[1],"values":[{"kind":"float","value":"nan"},-1.0]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::0f2efc98429606c7","dtype":"float16","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[1],"values":[1.0,1.0]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default","dtype":"float32","group":"movement","overload":"aten.select_scatter.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[0,1],"values":[[0,0,0],[0,0,0]]},{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[0],"values":[1,1]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::0fa877bf350d423a","dtype":"int32","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808]]},{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[1],"values":[-9223372036854775808,-1]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::381869f6ec8d522f","dtype":"int64","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-128,-1,0],[1,127,-128]]},{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2],"storage_offset":0,"stride":[1],"values":[-128,-1]},1,1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.select_scatter.default::26a8cc601154c5b1","dtype":"int8","group":"movement","overload":"aten.select_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":0,"stride":[0,1],"values":[[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"}]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":0,"stride":[0,1],"values":[[{"kind":"float","value":"-inf"},-1.0],[{"kind":"float","value":"-inf"},-1.0]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::0e7605481f8f680c","dtype":"bfloat16","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":0,"stride":[0,1],"values":[[false,false,false,false,false],[false,false,false,false,false]]},{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":0,"stride":[0,1],"values":[[false,false],[false,false]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::14a244a45db6f900","dtype":"bool","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":0,"stride":[10,2],"values":[[{"imag":0.0,"kind":"complex","real":-0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":-0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":-0.0}],[{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":-0.0},{"imag":0.0,"kind":"complex","real":0.0},{"imag":0.0,"kind":"complex","real":-0.0},{"imag":0.0,"kind":"complex","real":0.0}]]},{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":0,"stride":[4,2],"values":[[{"imag":0.0,"kind":"complex","real":-0.0},{"imag":0.0,"kind":"complex","real":0.0}],[{"imag":0.0,"kind":"complex","real":-0.0},{"imag":0.0,"kind":"complex","real":0.0}]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::416ac198a93fda4b","dtype":"complex64","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":0,"stride":[10,2],"values":[[-2.0,-1.0,0.0,3.0,-2.0],[-1.0,0.0,3.0,-2.0,-1.0]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":0,"stride":[4,2],"values":[[-2.0,-1.0],[0.0,3.0]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::0fb3bf3225ae0f48","dtype":"float16","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":0,"stride":[5,1],"values":[[-0.75,-0.5277777910232544,-0.3055555522441864,-0.0833333283662796,0.1388888955116272],[0.3611111044883728,0.5833333134651184,0.8055555820465088,1.0277777910232544,1.25]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":0,"stride":[2,1],"values":[[1.0,1.0],[1.0,1.0]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default","dtype":"float32","group":"movement","overload":"aten.slice_scatter.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":0,"stride":[5,1],"values":[[0,0,0,0,0],[0,0,0,1,1]]},{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":0,"stride":[2,1],"values":[[1,1],[1,1]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::66513922f364f74c","dtype":"int32","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":1,"stride":[5,1],"values":[[0,0,0,0,0],[0,0,0,0,0]]},{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":1,"stride":[2,1],"values":[[0,0],[0,0]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::3052a37b934f662e","dtype":"int64","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,5],"storage_offset":1,"stride":[5,1],"values":[[0,0,0,0,0],[0,0,0,0,0]]},{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,2],"storage_offset":1,"stride":[2,1],"values":[[0,0],[0,0]]},1,1,5,2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.slice_scatter.default::04062992c91e7941","dtype":"int8","group":"movement","overload":"aten.slice_scatter.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":1,"stride":[3,3,1],"values":[[[-0.0,0.0,-0.0]],[[0.0,-0.0,0.0]]]},0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::02b48cfa52985f31","dtype":"bfloat16","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":0,"stride":[3,3,1],"values":[[[true,true,true]],[[true,true,true]]]},2],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::1bcb9e4c94d596ee","dtype":"bool","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":0,"stride":[3,3,1],"values":[[[{"imag":0.0,"kind":"complex","real":-0.75},{"imag":0.0,"kind":"complex","real":-0.3499999940395355},{"imag":0.0,"kind":"complex","real":0.050000011920928955}]],[[{"imag":0.0,"kind":"complex","real":0.44999998807907104},{"imag":0.0,"kind":"complex","real":0.8500000238418579},{"imag":0.0,"kind":"complex","real":1.25}]]]},1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::103e65bd47ef31b5","dtype":"complex64","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":0,"stride":[3,3,1],"values":[[[-0.75,-0.35009765625,0.04998779296875]],[[0.449951171875,0.85009765625,1.25]]]},1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::0560919b44e4d9b9","dtype":"float16","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":0,"stride":[3,3,1],"values":[[[-0.75,-0.3499999940395355,0.050000011920928955]],[[0.44999998807907104,0.8500000238418579,1.25]]]},1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim","dtype":"float32","group":"movement","overload":"aten.squeeze.dim","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":1,"stride":[3,3,1],"values":[[[-2,-1,0]],[[3,-2,-1]]]},0],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::0c574afedb9546d5","dtype":"int32","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":0,"stride":[0,3,1],"values":[[[0,0,0]],[[0,0,0]]]},1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::2e71d6797efe4339","dtype":"int64","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[2,1,3],"storage_offset":0,"stride":[0,3,1],"values":[[[0,0,0]],[[0,0,0]]]},1],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dim::2b317b078eeca410","dtype":"int8","group":"movement","overload":"aten.squeeze.dim","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,1,6,2],"values":[[[[{"kind":"float","value":"nan"},-1.0,0.0]],[[1.0,{"kind":"float","value":"nan"},-1.0]]]]},[]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::09c1ab14880d7472","dtype":"bfloat16","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":1,"stride":[6,3,3,1],"values":[[[[false,false,false]],[[false,false,false]]]]},[-1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::18779e7dac86a081","dtype":"bool","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"complex64","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,0,3,1],"values":[[[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}]],[[{"imag":0.0,"kind":"complex","real":{"kind":"float","value":"nan"}},{"imag":0.0,"kind":"complex","real":-1.0},{"imag":0.0,"kind":"complex","real":0.0}]]]]},[0,3]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::284cbcb62c5ee4c3","dtype":"complex64","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,3,3,1],"values":[[[[{"kind":"float","value":"-inf"},-1.0,0.0]],[[{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0]]]]},[0,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::2f829d272e698e81","dtype":"float16","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,3,3,1],"values":[[[[-0.75,-0.3499999940395355,0.050000011920928955]],[[0.44999998807907104,0.8500000238418579,1.25]]]]},[0,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims","dtype":"float32","group":"movement","overload":"aten.squeeze.dims","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,0,3,1],"values":[[[[0,0,0]],[[0,0,0]]]]},[-1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::235df64f9d90e0d0","dtype":"int32","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,1,6,2],"values":[[[[-2,-1,0]],[[3,-2,-1]]]]},[]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::0beec3ddd883f085","dtype":"int64","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[1,2,1,3],"storage_offset":0,"stride":[6,0,3,1],"values":[[[[0,0,0]],[[0,0,0]]]]},[-1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.squeeze.dims::0b9e4727dfd74c48","dtype":"int8","group":"movement","overload":"aten.squeeze.dims","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bool","kind":"tensor","requires_grad":false,"shape":[0],"storage_offset":0,"stride":[1],"values":[]},[0],false],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bool"}}},"case_id":"aten.sum.dim_IntList::f5051578a7740a72","dtype":"bool","group":"movement","overload":"aten.sum.dim_IntList","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,0,1],"values":[[[[{"kind":"float","value":"nan"},-1.0,0.0],[{"kind":"float","value":"nan"},-1.0,0.0],[{"kind":"float","value":"nan"},-1.0,0.0]]]]},[6,6],null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.upsample_nearest2d.vec::0b904b6c74475ffe","dtype":"bfloat16","group":"movement","overload":"aten.upsample_nearest2d.vec","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[18,18,6,2],"values":[[[[-2.0,-1.0,0.0],[3.0,-2.0,-1.0],[0.0,3.0,-2.0]]]]},[6,6],null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.upsample_nearest2d.vec::494159e218007566","dtype":"float16","group":"movement","overload":"aten.upsample_nearest2d.vec","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,3,3],"storage_offset":0,"stride":[9,9,3,1],"values":[[[[-0.75,-0.5,-0.25],[0.0,0.25,0.5],[0.75,1.0,1.25]]]]},[6,6],null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.upsample_nearest2d.vec","dtype":"float32","group":"movement","overload":"aten.upsample_nearest2d.vec","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,0,4,1],"values":[[[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]],[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]]]]},[2,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool2d.default::0c8bbb7c015089e4","dtype":"bfloat16","group":"reduction","overload":"aten._adaptive_avg_pool2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,0,4,1],"values":[[[[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0]],[[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0]]]]},[2,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool2d.default::090439209cb4cc6b","dtype":"float16","group":"reduction","overload":"aten._adaptive_avg_pool2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,16,4,1],"values":[[[[-0.75,-0.6854838728904724,-0.6209677457809448,-0.5564516186714172],[-0.49193549156188965,-0.42741936445236206,-0.3629032373428345,-0.2983871102333069],[-0.2338709831237793,-0.1693548560142517,-0.10483872890472412,-0.04032260179519653],[0.024193525314331055,0.08870965242385864,0.15322577953338623,0.21774190664291382]],[[0.2822580933570862,0.34677422046661377,0.41129034757614136,0.47580647468566895],[0.5403226017951965,0.6048387289047241,0.6693548560142517,0.7338709831237793],[0.7983871102333069,0.8629032373428345,0.9274193644523621,0.9919354915618896],[1.0564515590667725,1.1209677457809448,1.1854839324951172,1.25]]]]},[2,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool2d.default","dtype":"float32","group":"reduction","overload":"aten._adaptive_avg_pool2d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":1,"stride":[4,4,2,1],"values":[[[[-3.3895313892515355e+38,-1.0],[0.0,1.0]]]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":1,"stride":[16,16,4,1],"values":[[[[-3.3895313892515355e+38,-1.0,0.0,1.0],[3.3895313892515355e+38,-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38,-1.0],[0.0,1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool2d_backward.default::122fc8dae65853eb","dtype":"bfloat16","group":"reduction","overload":"aten._adaptive_avg_pool2d_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":0,"stride":[4,4,0,1],"values":[[[[-0.0,0.0],[-0.0,0.0]]]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":0,"stride":[16,16,0,1],"values":[[[[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0],[-0.0,0.0,-0.0,0.0]]]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool2d_backward.default::15eb1c963524b92b","dtype":"float16","group":"reduction","overload":"aten._adaptive_avg_pool2d_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":0,"stride":[4,4,2,1],"values":[[[[1.0,1.0],[1.0,1.0]]]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":0,"stride":[16,16,4,1],"values":[[[[-0.75,-0.6166666746139526,-0.4833333194255829,-0.34999996423721313],[-0.21666663885116577,-0.08333329856395721,0.05000004172325134,0.1833333820104599],[0.3166666030883789,0.44999995827674866,0.5833333134651184,0.7166666388511658],[0.8499999642372131,0.9833333492279053,1.1166666746139526,1.25]]]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool2d_backward.default","dtype":"float32","group":"reduction","overload":"aten._adaptive_avg_pool2d_backward.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,4,4,4],"storage_offset":0,"stride":[128,128,32,8,2],"values":[[[[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]]]]},[2,2,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool3d.default::25d3903dddc62516","dtype":"bfloat16","group":"reduction","overload":"aten._adaptive_avg_pool3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,4,4,4],"storage_offset":1,"stride":[64,64,16,4,1],"values":[[[[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]],[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]],[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]],[[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0],[-2.0,-1.0,0.0,3.0]]]]]},[2,2,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool3d.default::01991c99b1bc6f70","dtype":"float16","group":"reduction","overload":"aten._adaptive_avg_pool3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,4,4,4],"storage_offset":0,"stride":[64,64,16,4,1],"values":[[[[[-0.75,-0.7182539701461792,-0.6865079402923584,-0.6547619104385376],[-0.6230158805847168,-0.591269850730896,-0.5595238208770752,-0.5277777910232544],[-0.4960317313671112,-0.4642857015132904,-0.4325396716594696,-0.4007936418056488],[-0.3690475821495056,-0.3373015522956848,-0.305555522441864,-0.2738094925880432]],[[-0.2420634627342224,-0.2103174328804016,-0.1785714030265808,-0.14682535827159882],[-0.11507932841777802,-0.08333329856395721,-0.05158726125955582,-0.019841227680444717],[0.011904805898666382,0.04365083947777748,0.07539687305688858,0.10714291036128998],[0.13888894021511078,0.17063497006893158,0.20238101482391357,0.23412704467773438]],[[0.2658729553222656,0.2976189851760864,0.3293650150299072,0.361111044883728],[0.3928571045398712,0.424603134393692,0.4563491642475128,0.4880951941013336],[0.5198412537574768,0.5515872836112976,0.5833333134651184,0.6150793433189392],[0.64682537317276,0.6785714030265808,0.7103174328804016,0.7420634627342224]],[[0.7738094925880432,0.805555522441864,0.8373015522956848,0.8690475821495056],[0.9007936120033264,0.9325396418571472,0.964285671710968,0.9960317611694336],[1.0277777910232544,1.0595238208770752,1.091269850730896,1.1230158805847168],[1.1547619104385376,1.1865079402923584,1.2182539701461792,1.25]]]]]},[2,2,2]],"kind":"tuple"},"kwargs":{}},"case_id":"aten._adaptive_avg_pool3d.default","dtype":"float32","group":"reduction","overload":"aten._adaptive_avg_pool3d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":0,"stride":[12,0,1],"values":[[[-3.3895313892515355e+38,-1.0,0.0,1.0,3.3895313892515355e+38,-3.3895313892515355e+38],[-3.3895313892515355e+38,-1.0,0.0,1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]]},[3]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.adaptive_avg_pool1d.default::0918a4c37987438c","dtype":"bfloat16","group":"reduction","overload":"aten.adaptive_avg_pool1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":1,"stride":[12,6,1],"values":[[[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0],[0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}]]]},[3]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.adaptive_avg_pool1d.default::0c908097ebb736f6","dtype":"float16","group":"reduction","overload":"aten.adaptive_avg_pool1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":0,"stride":[12,6,1],"values":[[[-0.75,-0.5681818127632141,-0.3863636255264282,-0.20454543828964233,-0.022727251052856445,0.15909093618392944],[0.34090906381607056,0.5227272510528564,0.7045454382896423,0.8863636255264282,1.0681817531585693,1.25]]]},[3]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.adaptive_avg_pool1d.default","dtype":"float32","group":"reduction","overload":"aten.adaptive_avg_pool1d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":1,"stride":[12,6,1],"values":[[[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0],[0.0,{"kind":"float","value":"inf"},{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}]]]},[2],[2],[0],false,true],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool1d.default::13b645388bc785ee","dtype":"bfloat16","group":"reduction","overload":"aten.avg_pool1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":1,"stride":[12,6,1],"values":[[[-2.0,-1.0,0.0,3.0,-2.0,-1.0],[0.0,3.0,-2.0,-1.0,0.0,3.0]]]},[2],[2],[0],false,true],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool1d.default::1226d87080a1ee0b","dtype":"float16","group":"reduction","overload":"aten.avg_pool1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":0,"stride":[12,6,1],"values":[[[-0.75,-0.5681818127632141,-0.3863636255264282,-0.20454543828964233,-0.022727251052856445,0.15909093618392944],[0.34090906381607056,0.5227272510528564,0.7045454382896423,0.8863636255264282,1.0681817531585693,1.25]]]},[2],[2],[0],false,true],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool1d.default","dtype":"float32","group":"reduction","overload":"aten.avg_pool1d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,2,6],"storage_offset":0,"stride":[24,12,2],"values":[[[-2,-1,0,3,-2,-1],[0,3,-2,-1,0,3]]]},[2],[2],[0],true,false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool1d.default::274a3539eb811977","dtype":"int64","group":"reduction","overload":"aten.avg_pool1d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,16,4,1],"values":[[[[-0.75,-0.68359375,-0.62109375,-0.5546875],[-0.4921875,-0.427734375,-0.36328125,-0.298828125],[-0.2333984375,-0.1689453125,-0.10498046875,-0.040283203125],[0.024169921875,0.0888671875,0.1533203125,0.2177734375]],[[0.283203125,0.34765625,0.412109375,0.4765625],[0.5390625,0.60546875,0.66796875,0.734375],[0.796875,0.86328125,0.92578125,0.9921875],[1.0546875,1.1171875,1.1875,1.25]]]]},[2,2],[2,2],[0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d.default::41e8d66633647c02","dtype":"bfloat16","group":"reduction","overload":"aten.avg_pool2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,0,4,1],"values":[[[[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}]],[[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}]]]]},[2,2],[2,2],[0,0],true,false,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d.default::06b1ccd3a8d452b6","dtype":"float16","group":"reduction","overload":"aten.avg_pool2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,16,4,1],"values":[[[[-0.75,-0.6854838728904724,-0.6209677457809448,-0.5564516186714172],[-0.49193549156188965,-0.42741936445236206,-0.3629032373428345,-0.2983871102333069],[-0.2338709831237793,-0.1693548560142517,-0.10483872890472412,-0.04032260179519653],[0.024193525314331055,0.08870965242385864,0.15322577953338623,0.21774190664291382]],[[0.2822580933570862,0.34677422046661377,0.41129034757614136,0.47580647468566895],[0.5403226017951965,0.6048387289047241,0.6693548560142517,0.7338709831237793],[0.7983871102333069,0.8629032373428345,0.9274193644523621,0.9919354915618896],[1.0564515590667725,1.1209677457809448,1.1854839324951172,1.25]]]]},[2,2],[2,2],[0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d.default","dtype":"float32","group":"reduction","overload":"aten.avg_pool2d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,2,4,4],"storage_offset":0,"stride":[32,16,4,1],"values":[[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]]]]},[2,2],[2,2],[0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d.default::0458e5f378acbd92","dtype":"int64","group":"reduction","overload":"aten.avg_pool2d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":0,"stride":[4,4,2,1],"values":[[[[-3.3895313892515355e+38,-1.0],[0.0,1.0]]]]},{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":0,"stride":[16,16,4,1],"values":[[[[-3.3895313892515355e+38,-1.0,0.0,1.0],[3.3895313892515355e+38,-3.3895313892515355e+38,-1.0,0.0],[1.0,3.3895313892515355e+38,-3.3895313892515355e+38,-1.0],[0.0,1.0,3.3895313892515355e+38,-3.3895313892515355e+38]]]]},[2,2],[2,2],[0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d_backward.default::0307a4878e7dad56","dtype":"bfloat16","group":"reduction","overload":"aten.avg_pool2d_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":0,"stride":[4,4,2,1],"values":[[[[1.0,1.0],[1.0,1.0]]]]},{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":0,"stride":[16,16,4,1],"values":[[[[-0.75,-0.61669921875,-0.4833984375,-0.35009765625],[-0.2166748046875,-0.08331298828125,0.04998779296875,0.183349609375],[0.316650390625,0.449951171875,0.58349609375,0.716796875],[0.85009765625,0.9833984375,1.1162109375,1.25]]]]},[2,2],[2,2],[0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d_backward.default::00c252656ce51458","dtype":"float16","group":"reduction","overload":"aten.avg_pool2d_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":0,"stride":[4,4,2,1],"values":[[[[1.0,1.0],[1.0,1.0]]]]},{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":0,"stride":[16,16,4,1],"values":[[[[-0.75,-0.6166666746139526,-0.4833333194255829,-0.34999996423721313],[-0.21666663885116577,-0.08333329856395721,0.05000004172325134,0.1833333820104599],[0.3166666030883789,0.44999995827674866,0.5833333134651184,0.7166666388511658],[0.8499999642372131,0.9833333492279053,1.1166666746139526,1.25]]]]},[2,2],[2,2],[0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d_backward.default","dtype":"float32","group":"reduction","overload":"aten.avg_pool2d_backward.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,2,2],"storage_offset":0,"stride":[4,4,0,1],"values":[[[[0,0],[0,0]]]]},{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,4,4],"storage_offset":0,"stride":[16,16,0,1],"values":[[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]]},[2,2],[2,2],[0,0],true,false,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool2d_backward.default::09257394e3803d86","dtype":"int64","group":"reduction","overload":"aten.avg_pool2d_backward.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,1,4,4,4],"storage_offset":0,"stride":[64,64,16,4,1],"values":[[[[[-0.75,-0.7182539701461792,-0.6865079402923584,-0.6547619104385376],[-0.6230158805847168,-0.591269850730896,-0.5595238208770752,-0.5277777910232544],[-0.4960317313671112,-0.4642857015132904,-0.4325396716594696,-0.4007936418056488],[-0.3690475821495056,-0.3373015522956848,-0.305555522441864,-0.2738094925880432]],[[-0.2420634627342224,-0.2103174328804016,-0.1785714030265808,-0.14682535827159882],[-0.11507932841777802,-0.08333329856395721,-0.05158726125955582,-0.019841227680444717],[0.011904805898666382,0.04365083947777748,0.07539687305688858,0.10714291036128998],[0.13888894021511078,0.17063497006893158,0.20238101482391357,0.23412704467773438]],[[0.2658729553222656,0.2976189851760864,0.3293650150299072,0.361111044883728],[0.3928571045398712,0.424603134393692,0.4563491642475128,0.4880951941013336],[0.5198412537574768,0.5515872836112976,0.5833333134651184,0.6150793433189392],[0.64682537317276,0.6785714030265808,0.7103174328804016,0.7420634627342224]],[[0.7738094925880432,0.805555522441864,0.8373015522956848,0.8690475821495056],[0.9007936120033264,0.9325396418571472,0.964285671710968,0.9960317611694336],[1.0277777910232544,1.0595238208770752,1.091269850730896,1.1230158805847168],[1.1547619104385376,1.1865079402923584,1.2182539701461792,1.25]]]]]},[2,2,2],[2,2,2],[0,0,0],false,true,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool3d.default","dtype":"float32","group":"reduction","overload":"aten.avg_pool3d.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[1,1,4,4,4],"storage_offset":0,"stride":[64,64,0,4,1],"values":[[[[[-9223372036854775808,-1,0,1],[9223372036854775807,-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808,-1],[0,1,9223372036854775807,-9223372036854775808]],[[-9223372036854775808,-1,0,1],[9223372036854775807,-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808,-1],[0,1,9223372036854775807,-9223372036854775808]],[[-9223372036854775808,-1,0,1],[9223372036854775807,-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808,-1],[0,1,9223372036854775807,-9223372036854775808]],[[-9223372036854775808,-1,0,1],[9223372036854775807,-9223372036854775808,-1,0],[1,9223372036854775807,-9223372036854775808,-1],[0,1,9223372036854775807,-9223372036854775808]]]]]},[2,2,2],[2,2,2],[0,0,0],true,false,null],"kind":"tuple"},"kwargs":{}},"case_id":"aten.avg_pool3d.default::12dcf423e90bb8de","dtype":"int64","group":"reduction","overload":"aten.avg_pool3d.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1,4,4],"storage_offset":0,"stride":[16,0,1],"values":[[[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}],[{"kind":"float","value":"-inf"},-1.0,0.0,{"kind":"float","value":"inf"}]]]},[3,3],[2,2],[1,1],[0,0],[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.col2im.default::19cacc859976f3ff","dtype":"bfloat16","group":"reduction","overload":"aten.col2im.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[1,4,4],"storage_offset":0,"stride":[32,8,2],"values":[[[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0],[{"kind":"float","value":"nan"},-1.0,0.0,1.0]]]},[3,3],[2,2],[1,1],[0,0],[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.col2im.default::08fdc8717c729bae","dtype":"float16","group":"reduction","overload":"aten.col2im.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[1,4,4],"storage_offset":0,"stride":[16,4,1],"values":[[[-0.75,-0.6166666746139526,-0.4833333194255829,-0.34999996423721313],[-0.21666663885116577,-0.08333329856395721,0.05000004172325134,0.1833333820104599],[0.3166666030883789,0.44999995827674866,0.5833333134651184,0.7166666388511658],[0.8499999642372131,0.9833333492279053,1.1166666746139526,1.25]]]},[3,3],[2,2],[1,1],[0,0],[1,1]],"kind":"tuple"},"kwargs":{}},"case_id":"aten.col2im.default","dtype":"float32","group":"reduction","overload":"aten.col2im.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[1],"storage_offset":1,"stride":[1],"values":[{"kind":"float","value":"nan"}]}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten.mean.default::3ef66692a4dd44da","dtype":"bfloat16","group":"reduction","overload":"aten.mean.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[],"storage_offset":0,"stride":[],"values":-0.75}],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten.mean.default::46876c7e49e46b14","dtype":"float16","group":"reduction","overload":"aten.mean.default","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]}],"kind":"tuple"},"kwargs":{}},"case_id":"aten.mean.default","dtype":"float32","group":"reduction","overload":"aten.mean.default","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"bfloat16","kind":"tensor","requires_grad":false,"shape":[0],"storage_offset":0,"stride":[1],"values":[]},[0],false],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"bfloat16"}}},"case_id":"aten.sum.dim_IntList::751948c6bf1ab7c9","dtype":"bfloat16","group":"reduction","overload":"aten.sum.dim_IntList","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float16","kind":"tensor","requires_grad":false,"shape":[0],"storage_offset":0,"stride":[1],"values":[]},[0],false],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"float16"}}},"case_id":"aten.sum.dim_IntList::f64516a97a0078fb","dtype":"float16","group":"reduction","overload":"aten.sum.dim_IntList","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"float32","kind":"tensor","requires_grad":false,"shape":[2,3],"storage_offset":0,"stride":[3,1],"values":[[-0.75,-0.3499999940395355,0.050000011920928955],[0.44999998807907104,0.8500000238418579,1.25]]},[0],false],"kind":"tuple"},"kwargs":{}},"case_id":"aten.sum.dim_IntList","dtype":"float32","group":"reduction","overload":"aten.sum.dim_IntList","source":"corpus_193"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int32","kind":"tensor","requires_grad":false,"shape":[0],"storage_offset":0,"stride":[1],"values":[]},null,false],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int32"}}},"case_id":"aten.sum.dim_IntList::21fc96f05f2d793e","dtype":"int32","group":"reduction","overload":"aten.sum.dim_IntList","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int64","kind":"tensor","requires_grad":false,"shape":[0],"storage_offset":0,"stride":[1],"values":[]},[],false],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int64"}}},"case_id":"aten.sum.dim_IntList::de708b47ebcdc321","dtype":"int64","group":"reduction","overload":"aten.sum.dim_IntList","source":"bounded"}'
    ),
    json.loads(
        '{"arguments":{"args":{"items":[{"dtype":"int8","kind":"tensor","requires_grad":false,"shape":[0],"storage_offset":0,"stride":[1],"values":[]},[-1],true],"kind":"tuple"},"kwargs":{"dtype":{"kind":"dtype","value":"int8"}}},"case_id":"aten.sum.dim_IntList::0374b5decc3eb91d","dtype":"int8","group":"reduction","overload":"aten.sum.dim_IntList","source":"bounded"}'
    ),
]


def _decode(value, inputs=()):
    if isinstance(value, list):
        return [_decode(item, inputs) for item in value]
    if not isinstance(value, dict):
        return value
    if "input_index" in value:
        return inputs[value["input_index"]]
    kind = value.get("kind")
    if kind == "tuple":
        return tuple(_decode(item, inputs) for item in value["items"])
    if kind == "dtype" or kind == "memory_format":
        return getattr(torch, value["value"])
    if kind == "device":
        return torch.device(value["value"])
    if kind == "complex":
        return complex(_decode(value["real"]), _decode(value["imag"]))
    if kind == "float":
        return float(value["value"])
    if kind == "tensor":
        dtype = getattr(torch, value["dtype"])
        logical = torch.tensor(_decode(value["values"]), dtype=dtype).reshape(value["shape"])
        shape, stride = value["shape"], value["stride"]
        offset = value["storage_offset"]
        span = offset + (1 + sum((n - 1) * s for n, s in zip(shape, stride)) if all(shape) else 0)
        storage = torch.zeros(max(span, value.get("minimum_storage_elements", 0)), dtype=dtype)
        tensor = storage.as_strided(shape, stride, offset)
        if torch._debug_has_internal_overlap(tensor) == 0:
            tensor.copy_(logical)
        else:
            # Expanded inputs require scalar writes. Repeated physical addresses
            # must carry identical recorded bytes, including signed zero and NaN.
            seen = {}
            for index in itertools.product(*(range(extent) for extent in shape)):
                address = offset + sum(i * step for i, step in zip(index, stride))
                scalar = logical[index]
                if address in seen:
                    previous = seen[address].contiguous().reshape(-1).view(torch.uint8)
                    current = scalar.contiguous().reshape(-1).view(torch.uint8)
                    if not torch.equal(previous, current):
                        raise ValueError("recorded logical values are inconsistent with overlapping strides")
                seen[address] = scalar
                storage[address] = scalar
        return tensor.requires_grad_(value["requires_grad"])
    return {key: _decode(child, inputs) for key, child in value.items()}


def _op(name):
    namespace, packet, overload = name.split(".", 2)
    assert namespace == "aten"
    return getattr(getattr(torch.ops.aten, packet), overload)


# END GENERATED

PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"
GROUPS = tuple(sorted({row["group"] for row in ROWS}))


def validate_profile(profile):
    if (
        not isinstance(profile, dict)
        or not {"schema", "workload_id"} <= set(profile)
        or set(profile) - {"schema", "workload_id", "group"}
    ):
        raise ValueError("Core ATen coverage profile has an unsupported shape")
    if profile["schema"] != PROFILE_SCHEMA or profile["workload_id"] != "core_aten_coverage":
        raise ValueError("Core ATen coverage profile has an unsupported identity")
    if "group" not in profile:
        return {"groups": GROUPS}
    if not isinstance(profile["group"], str) or profile["group"] not in GROUPS:
        raise ValueError("Core ATen coverage profile has an invalid group")
    return {"groups": (profile["group"],)}


def _selected_profile():
    path = Path(__file__).with_name("profile.json")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Core ATen coverage profile must be an adjacent ordinary file")
    if not path.exists():
        return {"groups": GROUPS}, None
    raw = path.read_bytes()
    return validate_profile(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def _bind_inputs(value, inputs):
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            index = len(inputs)
            inputs.append(_decode(value))
            return {"input_index": index}
        return {key: _bind_inputs(child, inputs) for key, child in value.items()}
    if isinstance(value, list):
        return [_bind_inputs(child, inputs) for child in value]
    return value


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in _flatten(child)]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in _flatten(child)]
    if value is None:
        return []
    return [(value if isinstance(value, torch.Tensor) else torch.scalar_tensor(value)).reshape(-1)]


class CoreAtenCoverage(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, *inputs):
        outputs = []
        for operation, arguments in self.calls:
            decoded = _decode(arguments, inputs)
            outputs.extend(_flatten(operation(*decoded["args"], **decoded["kwargs"])))
        return tuple(outputs)


def get_model_and_inputs():
    profile, profile_sha256 = _selected_profile()
    inputs, calls = [], []
    for row in ROWS:
        if row["group"] in profile["groups"]:
            calls.append((_op(row["overload"]), _bind_inputs(row["arguments"], inputs)))
    model = CoreAtenCoverage(calls).eval()
    if profile_sha256 is not None:
        model.session_provenance = {
            "workload_id": "core_aten_coverage",
            "workload_role": "iteration",
            "input_source": "core_aten_corpus",
            "profile_sha256": profile_sha256,
        }
    return model, tuple(inputs)
