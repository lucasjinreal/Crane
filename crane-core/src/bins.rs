use anyhow::anyhow;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::fs::File;
use std::io::Read;

/**
 * Code for loading tensor data from python
 * it reads tensor by order:
 * [tensor1(dtype, shape_len, shape, data), tensor2, tensor3]
 */

pub fn load_tensors(path: &str) -> Result<Vec<Tensor>> {
    let mut file = File::open(path)?;
    let mut tensors = Vec::new();

    loop {
        // 读取类型名称长度（2字节，小端）
        let mut dtype_len_buf = [0u8; 2];
        if file.read(&mut dtype_len_buf)? != 2 {
            break; // 文件正常结束
        }
        let dtype_len = u16::from_le_bytes(dtype_len_buf) as usize;

        // 读取类型名称
        let mut dtype_name_buf = vec![0u8; dtype_len];
        file.read_exact(&mut dtype_name_buf)?;
        let dtype_name = String::from_utf8(dtype_name_buf)
            .map_err(|e| anyhow!("Invalid dtype string: {}", e))?;

        // 解析数据类型（支持常见类型）
        let dtype = match dtype_name.as_str() {
            "float32" => DType::F32,
            "float64" => DType::F64,
            "float16" => DType::F16,
            "int32" => {
                return Err(anyhow!(
                    "int32 not supported in candle, Unsupported dtype: {}",
                    dtype_name
                ))
            }
            "int64" => DType::I64,
            "uint8" => DType::U8,
            "bool" => DType::U8, // 根据实际情况调整
            _ => return Err(anyhow!("Unsupported dtype: {}", dtype_name)),
        };

        // 读取维度数（4字节，小端）
        let mut ndim_buf = [0u8; 4];
        file.read_exact(&mut ndim_buf)?;
        let ndim = u32::from_le_bytes(ndim_buf) as usize;

        // 读取形状数组（每个维度8字节，小端）
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            file.read_exact(&mut dim_buf)?;
            let dim = i64::from_le_bytes(dim_buf);
            if dim < 0 {
                return Err(anyhow!("Invalid negative dimension: {}", dim));
            }
            shape.push(dim as usize);
        }

        // 计算数据大小
        let element_count = shape.iter().product::<usize>();
        let bytes_needed = element_count * dtype.size_in_bytes();

        // 读取原始数据
        let mut data_buf = vec![0u8; bytes_needed];
        file.read_exact(&mut data_buf)?;

        // 转换为Tensor（注意字节顺序处理）
        let tensor = Tensor::from_raw_buffer(&data_buf, dtype, &shape, &Device::Cpu)
            .map_err(|e| anyhow!("Tensor creation failed: {}", e))?;

        tensors.push(tensor);
    }

    Ok(tensors)
}
