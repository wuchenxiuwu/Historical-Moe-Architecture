# -*- coding: utf-8 -*-
"""
MoE模型分析模块
====================
提供模型解释性分析、性能评估和诊断工具，支持SHAP值计算、LIME解释、
"""
import os
import time
import re
import numpy as np
import torch
import shap
import lime
import lime.lime_text
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# 导入核心模块
from moe_core import MoEModel
from config.config_loader import load_config

# 初始化日志系统
logger = logging.getLogger(__name__)

# 全局NLP模型缓存
_nlp_models = {}

def _get_language_detector(nlp: Language, name: str = "language_detector") -> LanguageDetector:
    """获取语言检测器组件"""
    if name not in nlp.pipe_names:
        nlp.add_pipe(name, last=True)
    return nlp.get_pipe(name)

def _load_nlp_model(lang_code: str) -> Language:
    """
    加载指定语言的NLP模型，带缓存机制
    
    Args:
        lang_code: 语言代码，如"en"或"zh"
        
    Returns:
        加载的spaCy语言模型
    """
    global _nlp_models
    
    # 根据语言代码选择合适的模型
    model_map = {
        "en": "en_core_web_sm",
        "zh": "zh_core_web_sm"
    }
    
    # 默认为英文模型
    model_name = model_map.get(lang_code, "en_core_web_sm")
    
    # 检查缓存
    if model_name in _nlp_models:
        return _nlp_models[model_name]
    
    try:
        # 加载模型
        nlp = spacy.load(model_name)
        
        # 添加语言检测组件
        if "language_detector" not in nlp.pipe_names:
            from spacy_langdetect import LanguageDetector
            nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
            
        # 存入缓存
        _nlp_models[model_name] = nlp
        logger.info(f"Loaded and cached NLP model: {model_name}")
        return nlp
    except OSError:
        logger.error(f"NLP model {model_name} not found. Please install it first.")
        raise
    except Exception as e:
        logger.error(f"Error loading NLP model {model_name}: {str(e)}", exc_info=True)
        raise

class MoEModelAnalyzer:
    """MoE模型分析器，提供全面的模型解释性和性能分析功能"""
    
    def __init__(self, model: MoEModel, config_path: str = "config/model_config.yaml"):
        """
        初始化模型分析器
        
        Args:
            model: 已训练的MoE模型实例
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)
        self.analysis_config = self.config.get("analysis", {})
        
        # 模型和设备配置
        self.model = model
        self.model.eval()  # 设置为评估模式
        self.device = next(model.parameters()).device if model else torch.device("cpu")
        
        # 模型解释器缓存
        self.shap_explainer = None
        self.lime_explainer = None
        
        # 可视化配置
        self.visualization_dir = self.analysis_config.get("visualization_dir", "./visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # 解释性配置参数
        self.max_examples = self.analysis_config.get("max_examples", 100)
        self.sample_size = self.analysis_config.get("sample_size", 50)
        self.random_state = self.analysis_config.get("random_state", 42)
        
        # 分词器配置
        self.tokenizer_config = self.analysis_config.get("tokenizer", {})
        self.enable_lemmatization = self.tokenizer_config.get("enable_lemmatization", True)
        self.filter_stop_words = self.tokenizer_config.get("filter_stop_words", True)
        self.lowercase = self.tokenizer_config.get("lowercase", True)
        self.max_token_length = self.tokenizer_config.get("max_token_length", 50)
        
        # 初始化解释器
        self._initialize_explainers()
        
        logger.info(f"Initialized MoEModelAnalyzer with device: {self.device}")
        
    def _initialize_explainers(self) -> None:
        """初始化模型解释器(SHAP和LIME)"""
        # 初始化LIME文本解释器
        try:
            self.lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=self.config.model.get("class_names", ["class_0", "class_1"]),
                split_expression=self._tokenize_text,
                random_state=self.random_state,
                bow=False  # 不使用词袋模型，保留序列信息
            )
            logger.info("Initialized LIME text explainer")
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {str(e)}", exc_info=True)
            
    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理：清理和标准化文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
            
        # 移除特殊字符和多余空白
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s\p{P}]", "", text, flags=re.UNICODE)
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        高级文本分词函数，用于LIME解释器和模型分析
        
        支持多语言自动检测，执行文本清理、分词、词形还原和停用词过滤
        
        Args:
            text: 原始文本
            
        Returns:
            分词后的tokens列表
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text input for tokenization")
            return []
            
        try:
            start_time = time.time()
            
            # 文本预处理
            processed_text = self._preprocess_text(text)
            if not processed_text:
                return []
                
            # 加载默认NLP模型进行语言检测
            default_nlp = _load_nlp_model("en")
            doc = default_nlp(processed_text[:500])  # 仅使用前500字符进行语言检测
            lang_code = doc._.language["language"]
            logger.debug(f"Detected language: {lang_code} for text: {processed_text[:50]}...")
            
            # 根据检测到的语言加载相应的NLP模型
            nlp = _load_nlp_model(lang_code)
            doc = nlp(processed_text)
            
            # 执行分词和后处理
            tokens = []
            for token in doc:
                # 过滤停用词
                if self.filter_stop_words and token.is_stop:
                    continue
                    
                # 过滤标点符号
                if token.is_punct or token.is_space:
                    continue
                    
                # 过滤过长token
                if self.max_token_length and len(token.text) > self.max_token_length:
                    continue
                    
                # 获取词形还原或原始文本
                if self.enable_lemmatization and token.lemma_:
                    token_text = token.lemma_
                else:
                    token_text = token.text
                    
                # 转小写
                if self.lowercase:
                    token_text = token_text.lower()
                    
                tokens.append(token_text)
            
            # 记录分词性能
            logger.debug(
                f"Tokenization completed in {time.time() - start_time:.4f}s. "
                f"Original length: {len(text)}, Tokens generated: {len(tokens)}"
            )
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error during text tokenization: {str(e)}", exc_info=True)
            # 失败时回退到基本分词
            return self._fallback_tokenize(text)
    
    def _fallback_tokenize(self, text: str) -> List[str]:
        """
        分词失败时的回退机制
        
        Args:
            text: 原始文本
            
        Returns:
            基本分词后的tokens列表
        """
        try:
            logger.warning("Falling back to basic whitespace tokenization")
            return self._preprocess_text(text).split()
        except Exception:
            logger.error("Fallback tokenization also failed")
            return []
        
    def prepare_shap_explainer(self, background_data: torch.Tensor) -> None:
        """
        准备SHAP解释器
        
        Args:
            background_data: 用于背景分布的样本数据，形状为 [N, input_dim]
        """
        if self.shap_explainer is not None:
            logger.warning("SHAP explainer already initialized, reinitializing...")
            
        try:
            # 将背景数据移至设备
            background_data = background_data.to(self.device)
            
            # 创建SHAP内核解释器
            self.shap_explainer = shap.KernelExplainer(
                model=self._shap_model_wrapper,
                data=background_data,
                link="logit"
            )
            logger.info(f"Initialized SHAP KernelExplainer with background data size: {background_data.shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}", exc_info=True)
            raise
            
    def _shap_model_wrapper(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        SHAP兼容的模型包装器
        
        Args:
            x: 输入特征数组或张量
            
        Returns:
            模型输出概率的numpy数组
        """
        # 转换为张量并移至设备
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        # 模型推理
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # 转换为numpy数组并返回
        return probabilities.cpu().numpy()
        
    def explain_prediction_lime(self, text: str, label: Optional[int] = None, num_features: int = 10) -> Dict:
        """
        使用LIME解释单个预测结果
        
        Args:
            text: 输入文本
            label: 要解释的类别标签，None表示预测的类别
            num_features: 要显示的特征数量
            
        Returns:
            包含解释结果和可视化路径的字典
        """
        if not self.lime_explainer:
            raise RuntimeError("LIME explainer not initialized. Call _initialize_explainers() first.")
            
        try:
            start_time = time.time()
            
            # 生成解释
            explanation = self.lime_explainer.explain_instance(
                text_instance=text,
                classifier_fn=self._lime_model_wrapper,
                labels=[label] if label is not None else None,
                num_features=num_features,
                num_samples=self.sample_size
            )
            
            # 确定要解释的标签
            if label is None:
                # 获取预测的标签
                pred_probs = self._lime_model_wrapper([text])[0]
                label = pred_probs.argmax()
                
            # 保存可视化结果
            vis_path = os.path.join(self.visualization_dir, f"lime_explanation_label_{label}_{int(time.time())}.html")
            explanation.save_to_file(vis_path)
            
            # 提取解释特征
            explanation_features = explanation.as_list(label=label)
            
            logger.info(f"Generated LIME explanation for label {label} in {time.time()-start_time:.2f}s, saved to {vis_path}")
            
            return {
                "label": label,
                "explanation_features": explanation_features,
                "visualization_path": vis_path,
                "predicted_probability": float(self._lime_model_wrapper([text])[0][label])
            }
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}", exc_info=True)
            raise
            
    def _lime_model_wrapper(self, texts: List[str]) -> np.ndarray:
        """
        LIME兼容的模型包装器
        
        Args:
            texts: 文本列表
            
        Returns:
            模型输出概率的numpy数组，形状为 [n_samples, n_classes]
        """
        # 在实际应用中，这里应该将文本转换为模型输入特征
        # 这里使用随机特征作为示例
        input_features = np.random.randn(len(texts), self.config.model.get("input_dim", 512))
        input_tensor = torch.tensor(input_features, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor).cpu().numpy()
            probabilities = torch.nn.functional.softmax(torch.tensor(outputs), dim=1).numpy()
            
        return probabilities
        
    def explain_predictions_shap(self, input_features: torch.Tensor, max_evals: int = 200) -> Dict:
        """
        使用SHAP解释多个预测结果
        
        Args:
            input_features: 输入特征张量，形状为 [n_samples, input_dim]
            max_evals: SHAP计算的最大评估次数
            
        Returns:
            包含SHAP值和可视化路径的字典
        """
        if not self.shap_explainer:
            raise RuntimeError("SHAP explainer not initialized. Call prepare_shap_explainer() first.")
            
        try:
            start_time = time.time()
            input_features = input_features.cpu().numpy()  # SHAP在CPU上运行更稳定
            
            # 计算SHAP值
            shap_values = self.shap_explainer.shap_values(
                X=input_features,
                max_evals=max_evals,
                batch_size=min(16, len(input_features)),
                silent=True
            )
            
            # 保存摘要图可视化
            summary_vis_path = os.path.join(self.visualization_dir, f"shap_summary_{int(time.time())}.png")
            
            # 创建SHAP摘要图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                input_features, 
                feature_names=[f"feature_{i}" for i in range(input_features.shape[1])],
                show=False
            )
            plt.tight_layout()
            plt.savefig(summary_vis_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Generated SHAP explanations for {input_features.shape[0]} samples in {time.time()-start_time:.2f}s, saved to {summary_vis_path}")
            
            return {
                "shap_values": shap_values,
                "summary_visualization_path": summary_vis_path,
                "sample_count": input_features.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}", exc_info=True)
            raise
            
    def analyze_expert_usage(self, input_features: torch.Tensor, return_details: bool = False) -> Union[Dict, Tuple[Dict, List]]:
        """
        分析专家使用情况和路由行为
        
        Args:
            input_features: 输入特征张量，形状为 [n_samples, input_dim]
            return_details: 是否返回详细的专家使用记录
            
        Returns:
            专家使用统计信息，可选包含详细记录
        """
        try:
            start_time = time.time()
            input_features = input_features.to(self.device)
            
            # 获取专家路由信息
            with torch.no_grad():
                # 前向传播并获取路由权重
                _, expert_stats = self.model(input_features, return_expert_stats=True)
            
            # 解析专家统计信息
            expert_usage = expert_stats["expert_usage"].cpu().numpy()  # [n_samples, n_experts]
            expert_load = np.sum(expert_usage, axis=0)  # [n_experts]
            expert_load_percent = (expert_load / np.sum(expert_load)) * 100
            
            # 计算专家使用分布的熵值（衡量负载均衡程度）
            expert_entropy = -np.sum((expert_load_percent / 100) * np.log(expert_load_percent / 100 + 1e-10))
            
            # 找出使用最多和最少的专家
            busiest_expert = np.argmax(expert_load)
            least_busy_expert = np.argmin(expert_load)
            
            # 保存专家负载可视化
            load_vis_path = os.path.join(self.visualization_dir, f"expert_load_{int(time.time())}.png")
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=np.arange(len(expert_load)), y=expert_load_percent)
            plt.title(f"Expert Load Distribution (Entropy: {expert_entropy:.2f})")
            plt.xlabel("Expert Index")
            plt.ylabel("Load Percentage (%)")
            plt.axhline(y=100/len(expert_load), color='r', linestyle='--', label='Perfect Balance')
            plt.legend()
            plt.tight_layout()
            plt.savefig(load_vis_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # 准备结果统计
            result = {
                "expert_count": len(expert_load),
                "total_samples": input_features.shape[0],
                "load_distribution": expert_load_percent.tolist(),
                "entropy": float(expert_entropy),
                "busiest_expert": int(busiest_expert),
                "busiest_load_percent": float(expert_load_percent[busiest_expert]),
                "least_busy_expert": int(least_busy_expert),
                "least_busy_load_percent": float(expert_load_percent[least_busy_expert]),
                "load_visualization_path": load_vis_path,
                "analysis_time": float(time.time() - start_time)
            }
            
            logger.info(f"Completed expert usage analysis for {input_features.shape[0]} samples")
            
            if return_details:
                return result, expert_usage.tolist()
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing expert usage: {str(e)}", exc_info=True)
            raise
            
    def evaluate_model_performance(self, test_loader, save_report: bool = True) -> Dict:
        """
        全面评估模型性能
        
        Args:
            test_loader: 测试数据加载器
            save_report: 是否保存评估报告
            
        Returns:
            包含各类评估指标的字典
        """
        start_time = time.time()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Evaluation batch {batch_idx}/{len(test_loader)} completed")
        
        # 计算评估指标
        class_names = self.config.model.get("class_names", [f"class_{i}" for i in range(len(np.unique(all_targets)))])
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 计算AUC（多类情况下使用ovo策略）
        try:
            auc_score = roc_auc_score(
                all_targets, 
                all_probabilities, 
                multi_class='ovo', 
                average='weighted'
            )
        except Exception as e:
            logger.warning(f"Could not compute AUC score: {str(e)}")
            auc_score = None
        
        # 保存评估报告和混淆矩阵可视化
        result = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "auc_score": auc_score,
            "class_metrics": {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1_score": report[cls]["f1-score"],
                    "support": int(report[cls]["support"])
                } for cls in class_names if cls in report
            },
            "total_samples": len(all_targets),
            "evaluation_time": float(time.time() - start_time)
        }
        
        # 保存评估报告和混淆矩阵可视化
        if save_report:
            cm_vis_path = os.path.join(self.visualization_dir, f"confusion_matrix_{int(time.time())}.png")
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(cm_vis_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # 保存文本报告
            report_vis_path = os.path.join(self.visualization_dir, f"evaluation_report_{int(time.time())}.txt")
            with open(report_vis_path, "w") as f:
                f.write("Model Evaluation Report\n")
                f.write("="*50 + "\n\n")
                f.write(f"Total Samples: {len(all_targets)}\n")
                f.write(f"Evaluation Time: {result['evaluation_time']:.2f}s\n\n")
                f.write("Overall Metrics:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision (weighted): {result['precision']:.4f}\n")
                f.write(f"  Recall (weighted): {result['recall']:.4f}\n")
                f.write(f"  F1 Score (weighted): {result['f1_score']:.4f}\n")
                if auc_score is not None:
                    f.write(f"  AUC Score: {auc_score:.4f}\n\n")
                else:
                    f.write("  AUC Score: N/A\n\n")
                f.write("Class-wise Metrics:\n")
                for cls, metrics in result["class_metrics"].items():
                    f.write(f"  {cls}:\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F1 Score: {metrics['f1_score']:.4f}\n")
                    f.write(f"    Support: {metrics['support']}\n\n")
            
            result["confusion_matrix_path"] = cm_vis_path
            result["report_path"] = report_vis_path
            
            logger.info(f"Saved model evaluation report to {report_vis_path}")
        
        logger.info(f"Completed model performance evaluation with {len(all_targets)} samples")
        
        return result
        
    def generate_attention_visualization(self, input_features: torch.Tensor, sample_idx: int = 0) -> str:
        """
        生成注意力权重可视化
        
        Args:
            input_features: 输入特征张量，形状为 [n_samples, input_dim]
            sample_idx: 要可视化的样本索引
            
        Returns:
            可视化文件路径
        """
        try:
            input_tensor = input_features[sample_idx:sample_idx+1].to(self.device)
            
            with torch.no_grad():
                _, attention_weights = self.model(input_tensor, return_attention=True)
            
            # 保存注意力权重可视化
            vis_path = os.path.join(self.visualization_dir, f"attention_weights_sample_{sample_idx}_{int(time.time())}.png")
            
            # 创建注意力热图
            plt.figure(figsize=(12, 8))
            for layer_idx, attn in enumerate(attention_weights):
                attn_map = attn[0].cpu().numpy()  # [n_heads, seq_len, seq_len]
                
                # 对多头注意力取平均
                avg_attn = np.mean(attn_map, axis=0)
                
                plt.subplot(len(attention_weights), 1, layer_idx+1)
                sns.heatmap(avg_attn, cmap='viridis')
                plt.title(f"Attention Weights - Layer {layer_idx+1}")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Generated attention visualization for sample {sample_idx}, saved to {vis_path}")
            
            return vis_path
            
        except Exception as e:
            logger.error(f"Error generating attention visualization: {str(e)}", exc_info=True)
            raise
            
    def analyze_model_calibration(self, test_loader, num_bins: int = 10) -> Dict:
        """
        分析模型校准性能（预测置信度与实际准确率的匹配程度）
        
        Args:
            test_loader: 测试数据加载器
            num_bins: 置信度分箱数量
            
        Returns:
            包含校准指标的字典
        """
        all_confidences = []
        all_correct = []
        
        self.model.eval()
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                all_confidences.extend(confidences.cpu().numpy())
                all_correct.extend((predictions == targets).cpu().numpy())
        
        # 计算校准指标
        confidences = np.array(all_confidences)
        correct = np.array(all_correct)
        
        # 分箱计算准确率和置信度
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries, right=True) - 1
        
        bin_acc = np.zeros(num_bins)
        bin_conf = np.zeros(num_bins)
        bin_count = np.zeros(num_bins)
        
        for b in range(num_bins):
            bin_mask = bin_indices == b
            if np.sum(bin_mask) > 0:
                bin_acc[b] = np.mean(correct[bin_mask])
                bin_conf[b] = np.mean(confidences[bin_mask])
                bin_count[b] = np.sum(bin_mask)
        
        # 计算ECE (Expected Calibration Error)
        ece = np.sum((bin_count / len(confidences)) * np.abs(bin_acc - bin_conf))
        
        # 计算MCE (Maximum Calibration Error)
        mce = np.max(np.abs(bin_acc - bin_conf))
        
        # 保存校准曲线图
        calib_vis_path = os.path.join(self.visualization_dir, f"calibration_curve_{int(time.time())}.png")
        
        plt.figure(figsize=(10, 8))
        plt.bar(bin_boundaries[:-1], bin_acc, width=1/num_bins, alpha=0.5, label='Accuracy')
        plt.bar(bin_boundaries[:-1], bin_conf, width=1/num_bins, alpha=0.5, label='Confidence')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f"Calibration Curve (ECE: {ece:.4f}, MCE: {mce:.4f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(calib_vis_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        result = {
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "bin_accuracies": bin_acc.tolist(),
            "bin_confidences": bin_conf.tolist(),
            "bin_counts": bin_count.tolist(),
            "bin_boundaries": bin_boundaries.tolist(),
            "calibration_visualization_path": calib_vis_path
        }
        
        logger.info(f"Completed model calibration analysis, ECE: {ece:.4f}, MCE: {mce:.4f}")
        
        return result

class MoEPerformanceAnalyzer:
    """MoE模型性能分析器，提供训练和推理性能评估工具"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        初始化性能分析器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)
        self.perf_config = self.config.get("performance", {})
        
        # 性能分析配置
        self.warmup_runs = self.perf_config.get("warmup_runs", 5)
        self.benchmark_runs = self.perf_config.get("benchmark_runs", 20)
        self.input_shapes = self.perf_config.get("input_shapes", [
            (1, 512),   # 小批量小序列
            (16, 512),  # 中批量中序列
            (32, 1024)  # 大批量长序列
        ])
        
        # 结果保存路径
        self.perf_results_dir = self.perf_config.get("results_dir", "./performance_results")
        os.makedirs(self.perf_results_dir, exist_ok=True)
        
        logger.info("Initialized MoEPerformanceAnalyzer")
        
    def benchmark_inference_performance(self, model: MoEModel) -> Dict:
        """
        基准测试模型推理性能
        
        Args:
            model: 要测试的MoE模型
            
        Returns:
            包含性能指标的字典
        """
        model.eval()
        device = next(model.parameters()).device
        results = {}
        
        logger.info(f"Starting inference benchmark on {device}")
        
        with torch.no_grad():
            # 执行预热运行
            for _ in range(self.warmup_runs):
                input_tensor = torch.randn(1, self.config.model.get("input_dim", 512), device=device)
                model(input_tensor)
            
            # 对不同输入形状进行基准测试
            for shape in self.input_shapes:
                batch_size, seq_len = shape
                input_tensor = torch.randn(batch_size, seq_len, device=device)
                
                # 记录推理时间
                start_time = time.time()
                for _ in range(self.benchmark_runs):
                    model(input_tensor)
                    if device.type == "cuda":
                        torch.cuda.synchronize()  # 等待GPU操作完成
                
                total_time = time.time() - start_time
                avg_time = total_time / self.benchmark_runs
                throughput = (batch_size * self.benchmark_runs) / total_time
                
                # 记录结果
                results[f"shape_{batch_size}_{seq_len}"] = {
                    "batch_size": batch_size,
                    "sequence_length": seq_len,
                    "total_time": total_time,
                    "average_time_per_batch": avg_time,
                    "throughput_samples_per_sec": throughput,
                    "device": str(device)
                }
                
                logger.info(
                    f"Benchmark complete - Batch: {batch_size}, Seq Len: {seq_len}, "
                    f"Avg Time: {avg_time:.4f}s, Throughput: {throughput:.2f} samples/sec"
                )
        
        # 保存性能结果
        results_path = os.path.join(self.perf_results_dir, f"inference_benchmark_{int(time.time())}.json")
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        results["results_path"] = results_path
        logger.info(f"Saved inference benchmark results to {results_path}")
        
        return results

# 导出公共API
__all__ = ["MoEModelAnalyzer", "MoEPerformanceAnalyzer"]

logger.info("MoE analysis module loaded successfully")