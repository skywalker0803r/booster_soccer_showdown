# -*- coding: utf-8 -*-
# llm_coach.py
"""
LLM教練模組：策略層
根據訓練統計數據（跌倒率、移動距離），決定當前訓練階段（Phase）和獎勵權重（Weights）
集成 Google Gemini API 進行智能決策
"""

import numpy as np
import json
import time
from typing import Dict, Any, Optional

# Google Gemini API 整合
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ google-generativeai 未安裝，將使用啟發式規則。安裝指令: pip install google-generativeai")

class LLMCoach:
    def __init__(self, api_key: str = None, use_llm: bool = True):
        """
        初始化LLM教練
        
        Args:
            api_key: Google Gemini API key
            use_llm: 是否使用真正的LLM推理（False時使用啟發式規則）
        """
        # 初始權重 (Phase 1: 站立和生存)
        self.current_weights = {
            "balance": 1.0,   # 高度重視平衡和直立
            "progress": 0.0,  # 暫時忽略移動和進度
            "energy": 0.1     # 輕微節省能量懲罰
        }
        self.phase = "Stance"
        self.phase_history = []
        
        # LLM API 配置
        self.use_llm = use_llm and GENAI_AVAILABLE
        self.api_key = api_key
        self.model = None
        self.api_calls_count = 0
        self.api_errors_count = 0
        
        # 初始化 Gemini API
        if self.use_llm and api_key:
            try:
                genai.configure(api_key=api_key)
                # 使用最新穩定的 Gemini 模型
                self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                print("✅ Gemini API 已初始化，將使用真正的LLM推理")
            except Exception as e:
                print(f"⚠️ Gemini API 初始化失敗: {e}")
                print("🔄 將回退至啟發式規則")
                self.use_llm = False
        else:
            print("🤖 使用啟發式規則進行決策（未提供API key或模組不可用）")
    
    def _llm_reasoning(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 Gemini API 進行智能推理
        
        Args:
            stats: 訓練統計數據
            
        Returns:
            dict: 包含階段和權重的決策結果
        """
        if not self.use_llm or not self.model:
            return None
            
        try:
            self.api_calls_count += 1
            
            # 構建給 LLM 的提示詞
            # 確保數值是標量而不是 numpy array，避免 deprecation 警告
            def safe_float(value):
                if hasattr(value, 'item'):
                    return value.item()
                else:
                    return float(value)
            
            avg_steps = safe_float(stats.get('avg_steps', 0))
            avg_reward = safe_float(stats.get('avg_reward', 0))
            fall_rate = safe_float(stats.get('fall_rate', 1.0))
            
            prompt = f"""
你是一個強化學習教練，正在訓練一個足球機器人。你需要根據當前的訓練統計數據，決定訓練階段和獎勵權重。

你必須嚴格遵守以下格式要求：你的回應只能是**單一的JSON物件**，不能包含任何額外的解釋性文字、前言或後語。這個JSON物件必須包含 "phase" 和 "weights" 兩個鍵。

## 當前狀況
- 平均存活步數: {avg_steps:.1f}
- 平均獎勵: {avg_reward:.2f}  
- 跌倒率: {fall_rate:.3f}
- 當前階段: {self.phase}
- 當前權重: {self.current_weights}

## 可選的訓練階段
1. "Stance & Survival" - 學習站立和基本生存
2. "Basic Movement" - 學習基礎移動和平衡
3. "Dribbling & Skills" - 學習踢球和控球技能
4. "Advanced Tactics" - 高級策略和精進

## 權重說明
- balance: 平衡和穩定性 (0.0-2.0)
- progress: 移動和進度 (0.0-2.0)  
- energy: 能量效率 (0.0-0.1)

## 決策原則
- 如果跌倒率高(>0.7)或步數少(<30)，專注站立 (高balance, 低progress)
- 如果能站穩但移動差，專注移動 (中balance, 中progress)
- 如果移動穩定，專注技能 (低balance, 高progress)

請根據統計數據，選擇最適合的階段和權重配置。

你的回應**必須**是一個單一的JSON物件，且只包含該JSON物件。
JSON格式範例:
```json
{{"phase": "Basic Movement", "weights": {{"balance": 1.5, "progress": 0.3, "energy": 0.03}}}}
```

現在，請輸出你的決策JSON物件：
"""

            # 調用 Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # 嘗試解析 JSON 響應
            # 找到 JSON 部分
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)
                
                # 驗證決策格式
                if 'phase' in decision and 'weights' in decision:
                    # 確保權重在合理範圍內
                    weights = decision['weights']
                    weights['balance'] = max(0.0, min(2.0, weights.get('balance', 1.0)))
                    weights['progress'] = max(0.0, min(2.0, weights.get('progress', 0.5)))
                    weights['energy'] = max(0.0, min(0.1, weights.get('energy', 0.02)))
                    
                    print(f"🧠 LLM推理成功 (API調用 #{self.api_calls_count})")
                    print(f"   LLM建議: {decision['phase']}")
                    print(f"   原始回應: {response_text[:100]}...")
                    
                    return decision
                else:
                    print(f"⚠️ LLM回應格式不正確: {response_text[:100]}...")
                    
            else:
                print(f"⚠️ 無法解析LLM回應中的JSON: {response_text[:100]}...")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析錯誤: {e}")
            print(f"   回應內容: {response_text[:200]}...")
        except Exception as e:
            print(f"⚠️ LLM API調用失敗: {e}")
            self.api_errors_count += 1
            
        return None
    
    def consult(self, stats):
        """
        根據訓練統計決定當前階段和權重
        優先使用 LLM API 推理，失敗時回退到啟發式規則
        
        Args:
            stats (dict): {
                'avg_steps': 平均存活步數,
                'avg_reward': 平均回合獎勵,
                'fall_rate': 跌倒頻率 (0.0~1.0),
                'avg_ball_distance': 平均與球的距離 (可選)
            }
        
        Returns:
            dict: 更新後的獎勵權重
        """
        
        previous_phase = self.phase
        decision_source = "heuristic"
        
        # 🤖 優先嘗試 LLM 推理
        if self.use_llm:
            llm_decision = self._llm_reasoning(stats)
            if llm_decision:
                # LLM 推理成功
                self.phase = llm_decision['phase']
                self.current_weights = llm_decision['weights']
                decision_source = "llm"
            else:
                # LLM 推理失敗，回退到啟發式規則
                print(f"🔄 LLM推理失敗，回退到啟發式規則")
                decision_source = "heuristic_fallback"
        
        # 🧮 啟發式規則 (LLM 不可用或失敗時的後備方案)
        if decision_source.startswith("heuristic"):
            steps = stats.get('avg_steps', 0)
            fall_rate = stats.get('fall_rate', 1.0)
            avg_reward = stats.get('avg_reward', -np.inf)
            
            # 階段判斷邏輯
            if steps < 30 or fall_rate > 0.8:
                # 階段 1: 學習站立和生存 (存活步數少或跌倒率高)
                self.phase = "Stance & Survival"
                self.current_weights = {
                    "balance": 2.0,   # 非常重視平衡
                    "progress": 0.05, # 極少進度獎勵
                    "energy": 0.02    # 輕微能量懲罰
                }
                
            elif steps >= 30 and steps < 80 and fall_rate <= 0.6:
                # 階段 2: 學習基礎移動 (能站穩但還不能長期行動)
                self.phase = "Basic Movement"
                self.current_weights = {
                    "balance": 1.5,   # 仍然重視平衡
                    "progress": 0.3,  # 開始引入進度獎勵
                    "energy": 0.03    # 稍微增加能量約束
                }
                
            elif steps >= 80 and fall_rate <= 0.4:
                # 階段 3: 學習踢球和技能 (能穩定行走)
                self.phase = "Dribbling & Skills"
                self.current_weights = {
                    "balance": 0.8,   # 降低平衡權重
                    "progress": 1.2,  # 大幅加強移動和球控制
                    "energy": 0.01    # 最小化能量懲罰
                }
                
            else:
                # 階段 4: 精進和最佳化 (高級技能)
                self.phase = "Advanced Tactics"
                self.current_weights = {
                    "balance": 0.3,   # 最小平衡權重
                    "progress": 1.8,  # 最大化技能獎勵
                    "energy": 0.005   # 幾乎無能量懲罰
                }
        
        # 記錄階段變化
        if previous_phase != self.phase:
            self.phase_history.append({
                'from_phase': previous_phase,
                'to_phase': self.phase,
                'stats': stats.copy(),
                'new_weights': self.current_weights.copy(),
                'decision_source': decision_source
            })
            
            if decision_source == "llm":
                print(f"🧠 LLM智能決策：階段變化 [{previous_phase}] → [{self.phase}]")
            else:
                print(f"🧠 啟發式規則：階段變化 [{previous_phase}] → [{self.phase}]")
            
            # 安全地提取數值用於顯示
            def safe_float_local(value):
                if hasattr(value, 'item'):
                    return value.item()
                else:
                    return float(value)
                    
            steps = safe_float_local(stats.get('avg_steps', 0))
            fall_rate = safe_float_local(stats.get('fall_rate', 1.0))
            print(f"   觸發條件：步數={steps:.1f}, 跌倒率={fall_rate:.3f}")
            print(f"   新權重：{self.current_weights}")
            print(f"   決策來源：{decision_source}")
        
        return self.current_weights
    
    def get_phase_info(self):
        """
        獲取當前階段的詳細信息
        
        Returns:
            dict: 包含當前階段信息的字典
        """
        return {
            'current_phase': self.phase,
            'current_weights': self.current_weights.copy(),
            'phase_history': self.phase_history.copy(),
            'api_stats': {
                'api_calls_count': self.api_calls_count,
                'api_errors_count': self.api_errors_count,
                'use_llm': self.use_llm,
                'success_rate': (self.api_calls_count - self.api_errors_count) / max(1, self.api_calls_count)
            }
        }
    
    def get_api_statistics(self):
        """
        獲取 API 使用統計
        
        Returns:
            dict: API 統計信息
        """
        return {
            'total_calls': self.api_calls_count,
            'errors': self.api_errors_count,
            'success_rate': (self.api_calls_count - self.api_errors_count) / max(1, self.api_calls_count),
            'llm_enabled': self.use_llm,
            'api_available': GENAI_AVAILABLE
        }
    
    def reset_coach(self, keep_api_config=True):
        """
        重置教練到初始狀態（用於新的訓練會話）
        
        Args:
            keep_api_config: 是否保留 API 配置（預設為True）
        """
        if keep_api_config:
            # 保留 API 配置和統計
            api_key = self.api_key
            use_llm = self.use_llm
            model = self.model
            api_calls = self.api_calls_count
            api_errors = self.api_errors_count
            
            # 重置其他屬性
            self.current_weights = {
                "balance": 1.0,
                "progress": 0.0,
                "energy": 0.1
            }
            self.phase = "Stance"
            self.phase_history = []
            
            # 恢復 API 配置
            self.api_key = api_key
            self.use_llm = use_llm
            self.model = model
            self.api_calls_count = api_calls
            self.api_errors_count = api_errors
            
            print("🧠 LLM教練已重置（保留API配置和統計）")
        else:
            # 完全重置
            api_key = self.api_key if hasattr(self, 'api_key') else None
            use_llm = self.use_llm if hasattr(self, 'use_llm') else True
            self.__init__(api_key, use_llm)
            print("🧠 LLM教練已完全重置")