import json
import csv
from typing import List, Dict, Union, Any, Iterator, Callable, Optional
import yaml
from pydantic import BaseModel, ValidationError, Field
from collections import defaultdict

class DataConfig(BaseModel):
    
    required_field: str = Field(..., description="预训练阶段必须包含的字段")
    schema_validation: bool = Field(..., description="是否启用schema验证")
    field_types: Dict[str, str] = Field(..., description="字段类型定义")
    min_length: int = Field(..., description="文本最小长度")
    max_length: int = Field(..., description="文本最大长度")

class CSVConfig(BaseModel):
    """CSV数据配置模型"""
    required_columns: List[str] = Field(..., description="CSV必填列")
    delimiter: str = Field(..., description="CSV分隔符")
    quotechar: str = Field(..., description="CSV引号字符")
    column_types: Dict[str, str] = Field(..., description="列类型定义")
    min_lengths: Dict[str, int] = Field(..., description="列最小长度")
    max_lengths: Dict[str, int] = Field(..., description="列最大长度")

class UnifiedConfig(BaseModel):
    """统一数据配置模型"""
    jsonl: DataConfig = Field(..., description="JSONL数据配置")
    csv: CSVConfig = Field(..., description="CSV数据配置")

class DataLoader:
    """JSONL/CSV数据加载器"""
    
    def __init__(
        self, 
        file_path: str, 
        config_path: str = "config/model_config.yaml",
        preprocess_hook: Optional[Callable[[Dict], Dict]] = None
    ):
        self.file_path = file_path
        self.data = []
        self.config = self._load_config(config_path)
        self.preprocess_hook = preprocess_hook
        
        # 根据文件类型选择相应配置
        if file_path.endswith('.jsonl'):
            self.data_config = self.config.jsonl
        elif file_path.endswith('.csv'):
            self.data_config = self.config.csv
        else:
            raise ValueError("不支持的文件格式，仅支持JSONL和CSV")
    
    def _load_config(self, config_path: str) -> UnifiedConfig:
        """安全加载配置，严格验证配置完整性"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # 提取数据配置部分
            data_config = config_data.get('data', {})
            
            # 验证配置完整性
            if not data_config or 'jsonl' not in data_config or 'csv' not in data_config:
                raise ValueError("配置文件不完整，必须包含jsonl和csv配置")
            
            # 构建统一配置
            return UnifiedConfig(
                jsonl=DataConfig(**data_config['jsonl']),
                csv=CSVConfig(**data_config['csv'])
            )
        except Exception as e:
            raise RuntimeError(f"配置加载失败: {str(e)}") from e
    
    def load(self) -> List[Dict]:
        """加载并验证数据"""
        if self.file_path.endswith('.jsonl'):
            self.data = self._load_jsonl()
        elif self.file_path.endswith('.csv'):
            self.data = self._load_csv()
        else:
            raise ValueError("❌ 不支持的文件格式，仅支持JSONL和CSV")
        
        return self.data
    
    def stream_data(self) -> Iterator[Dict]:
        """流式数据加载接口"""
        if self.file_path.endswith('.jsonl'):
            return self._stream_jsonl()
        elif self.file_path.endswith('.csv'):
            return self._stream_csv()
        else:
            raise ValueError("❌ 不支持的文件格式，仅支持JSONL和CSV")
    
    def sample_data(self, n: int = 100) -> List[Dict]:
        """数据采样功能"""
        if self.file_path.endswith('.jsonl'):
            return self._sample_jsonl(n)
        elif self.file_path.endswith('.csv'):
            return self._sample_csv(n)
        else:
            raise ValueError("❌ 不支持的文件格式，仅支持JSONL和CSV")
    
    def _load_jsonl(self) -> List[Dict]:
        """企业级JSONL加载器"""
        data = []
        for item in self._stream_jsonl():
            data.append(item)
        return data
    
    def _stream_jsonl(self) -> Iterator[Dict]:
        """JSONL流式加载"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self._validate_jsonl_item(item, line_num)
                    
                    # 应用预处理钩子
                    if self.preprocess_hook:
                        item = self.preprocess_hook(item)
                        
                    yield item
                except (json.JSONDecodeError, ValidationError) as e:
                    print(f"❌ 数据错误 (行 {line_num}): {str(e)}，已跳过")
    
    def _sample_jsonl(self, n: int = 100) -> List[Dict]:
        """JSONL数据采样"""
        sample = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self._validate_jsonl_item(item, i+1)
                    
                    # 应用预处理钩子
                    if self.preprocess_hook:
                        item = self.preprocess_hook(item)
                        
                    sample.append(item)
                except Exception as e:
                    print(f"❌ 采样数据错误 (行 {i+1}): {str(e)}，已跳过")
        
        return sample
    
    def _validate_jsonl_item(self, item: Dict, line_num: int):
        """企业级JSONL数据验证"""
        # 必填字段检查
        if self.config.jsonl.schema_validation:
            if self.config.jsonl.required_field not in item:
                raise ValidationError(f"缺少必填字段: {self.config.jsonl.required_field}")
            
            # 字段类型验证
            field_value = item[self.config.jsonl.required_field]
            expected_type = self.config.jsonl.field_types.get(self.config.jsonl.required_field, "string")
            
            if expected_type == "string" and not isinstance(field_value, str):
                raise ValidationError(f"字段类型错误，应为字符串")
            elif expected_type == "number" and not isinstance(field_value, (int, float)):
                raise ValidationError(f"字段类型错误，应为数字")
            
            # 长度验证
            if isinstance(field_value, str):
                if len(field_value) < self.config.jsonl.min_length:
                    raise ValidationError(f"文本过短 ({len(field_value)}字符)，最小要求 {self.config.jsonl.min_length}")
                if len(field_value) > self.config.jsonl.max_length:
                    raise ValidationError(f"文本过长 ({len(field_value)}字符)，最大允许 {self.config.jsonl.max_length}")
    
    def _load_csv(self) -> List[Dict]:
        """企业级CSV加载器"""
        data = []
        for item in self._stream_csv():
            data.append(item)
        return data
    
    def _stream_csv(self) -> Iterator[Dict]:
        """CSV流式加载"""
        csv_config = self.config.csv
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(
                f,
                delimiter=csv_config.delimiter,
                quotechar=csv_config.quotechar
            )
            
            # 检查必填列
            missing_columns = [col for col in csv_config.required_columns if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"❌ CSV文件缺少必要列: {', '.join(missing_columns)}")
            
            for row_num, row in enumerate(reader, 2):  # 从第2行开始（标题行是1）
                try:
                    # 企业级数据清洗
                    cleaned_row = {}
                    for key, value in row.items():
                        cleaned_row[key.strip()] = value.strip() if isinstance(value, str) else value
                    
                    # CSV行验证
                    self._validate_csv_row(cleaned_row, row_num)
                    
                    # 应用预处理钩子
                    if self.preprocess_hook:
                        cleaned_row = self.preprocess_hook(cleaned_row)
                        
                    yield cleaned_row
                except Exception as e:
                    print(f"❌ CSV处理错误 (行 {row_num}): {str(e)}，已跳过")
    
    def _sample_csv(self, n: int = 100) -> List[Dict]:
        """CSV数据采样"""
        sample = []
        csv_config = self.config.csv
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(
                f,
                delimiter=csv_config.delimiter,
                quotechar=csv_config.quotechar
            )
            
            # 检查必填列
            missing_columns = [col for col in csv_config.required_columns if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"❌ CSV文件缺少必要列: {', '.join(missing_columns)}")
            
            for i, row in enumerate(reader):
                if i >= n:
                    break
                    
                try:
                    # 数据清洗
                    cleaned_row = {}
                    for key, value in row.items():
                        cleaned_row[key.strip()] = value.strip() if isinstance(value, str) else value
                    
                    # 验证行数据
                    self._validate_csv_row(cleaned_row, i+2)  # 行号从2开始
                    
                    # 应用预处理钩子
                    if self.preprocess_hook:
                        cleaned_row = self.preprocess_hook(cleaned_row)
                        
                    sample.append(cleaned_row)
                except Exception as e:
                    print(f"❌ CSV采样错误 (行 {i+2}): {str(e)}，已跳过")
        
        return sample
    
    def _validate_csv_row(self, row: Dict, row_num: int):
        """CSV行数据验证"""
        csv_config = self.config.csv
        
        # 必填列值检查
        for col in csv_config.required_columns:
            if col not in row or not str(row[col]).strip():
                raise ValidationError(f"列 '{col}' 值为空或缺失")
            
            # 类型验证
            expected_type = csv_config.column_types.get(col, "string")
            value = row[col]
            
            if expected_type == "number":
                try:
                    float(value)
                except ValueError:
                    raise ValidationError(f"列 '{col}' 值 '{value}' 不是有效数字")
            
            # 长度验证
            if col in csv_config.min_lengths:
                min_len = csv_config.min_lengths[col]
                if len(str(value)) < min_len:
                    raise ValidationError(f"列 '{col}' 过短 ({len(str(value))}字符)，最小要求 {min_len}")
            
            if col in csv_config.max_lengths:
                max_len = csv_config.max_lengths[col]
                if len(str(value)) > max_len:
                    raise ValidationError(f"列 '{col}' 过长 ({len(str(value))}字符)，最大允许 {max_len}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """数据统计"""
        if not self.data:
            # 如果数据未加载，则先加载
            self.load()
        
        stats = {
            'total_records': len(self.data),
            'valid_records': 0,
            'invalid_records': 0,
            'invalid_reasons': defaultdict(int),
            'field_distribution': defaultdict(int),
            'length_distribution': {
                'min': float('inf'),
                'max': 0,
                'sum': 0,
                'average': 0
            }
        }
        
        # 确定主字段
        if self.file_path.endswith('.jsonl'):
            main_field = self.config.jsonl.required_field
        else:  # CSV
            main_field = self.config.csv.required_columns[0] if self.config.csv.required_columns else None
        
        for item in self.data:
            try:
                # 根据文件类型进行验证
                if self.file_path.endswith('.jsonl'):
                    self._validate_jsonl_item(item, 0)
                else:
                    self._validate_csv_row(item, 0)
                
                stats['valid_records'] += 1
                
                # 字段分布统计
                for field in item.keys():
                    stats['field_distribution'][field] += 1
                
                # 主字段长度统计
                if main_field and main_field in item:
                    value = item[main_field]
                    value_length = len(str(value))
                    
                    stats['length_distribution']['min'] = min(stats['length_distribution']['min'], value_length)
                    stats['length_distribution']['max'] = max(stats['length_distribution']['max'], value_length)
                    stats['length_distribution']['sum'] += value_length
                    
            except ValidationError as e:
                stats['invalid_records'] += 1
                # 提取错误原因（取冒号前的部分作为分类）
                reason = str(e).split(":", 1)[0].strip()
                stats['invalid_reasons'][reason] += 1
            except Exception as e:
                stats['invalid_records'] += 1
                stats['invalid_reasons']['其他错误'] += 1
        
        # 计算平均长度
        if stats['valid_records'] > 0 and stats['length_distribution']['sum'] > 0:
            stats['length_distribution']['average'] = round(
                stats['length_distribution']['sum'] / stats['valid_records'], 2
            )
        else:
            stats['length_distribution']['average'] = 0
        
        # 转换defaultdict为普通dict以便序列化
        stats['field_distribution'] = dict(stats['field_distribution'])
        stats['invalid_reasons'] = dict(stats['invalid_reasons'])
        
        return stats

class EnterpriseDataLoader(DataLoader):
    """数据加载器扩展"""
    
    def validate_schema(self) -> bool:
        """全面验证数据schema"""
        if not self.data:
            self.load()
            
        try:
            for item in self.data:
                if self.file_path.endswith('.jsonl'):
                    self._validate_jsonl_item(item, 0)
                else:
                    self._validate_csv_row(item, 0)
            return True
        except ValidationError:
            return False
    
    def export_quality_report(self, report_path: str):
        """导出数据质量报告"""
        stats = self.get_statistics()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        return stats