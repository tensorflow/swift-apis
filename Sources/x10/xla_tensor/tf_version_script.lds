VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    *TF_*;
    *TFE_*;
    tf_*;
    *XLATensor_*;
    copyTensor*;
    destroy*;
    DefaultDevice;
    getAllDevices;
    getDefaultDevice;
    setReplicationDevices;
    getReplicationDevices;
    syncLiveTensorsForDevices;
    ComputeIndexingBoundsAndStrides;
    DeleteString;
    GetStringCStr;
    MaterializedTensor_getData;
    MaterializedTensor_getType;
    PrintMetrics;
    XLAShape_*;
    SetMatMulPrecision;
    fetchTensorShape;
    *swift_xla*;
    *g_trace_level*;
    *xla*primitive_util*;
    *xla*MakeSpatialPadding*;
    *xla*PaddingConfig*;
    *google*protobuf*internal*;

  # Hide everything else.
  local:
    *;
};
