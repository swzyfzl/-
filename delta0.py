"""

期权策略Risk Reversal
背景：结合近期政策，指出股市不应该大起大落，在慢牛预期下，写下这个夏普增强策略，捕捉由风险逆转溢价带来的隐含偏度溢价，
即通过卖出被高估的虚值看跌期权、同时买入相对低估的虚值看涨期权
策略核心：在持有标的的情况下,用risk reversal期权策略构建一个组合头寸delta为0的组合，每日不断调整持仓始终保持delta中性，
在牛市下投资者为获取看涨期权的风险降低收益而愿意支付更高价格，所以卖call收入的权利金高于买put支付的权利金，
在标的缓慢上涨的同时拿到时间价值损耗部分的收益。但要注意，在标的大幅波动时，有vega敞口。

"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta


# -----------------------------
# 全局常量
# -----------------------------
INDEX_MULTIPLIER = 100  # IO 股指期权合约乘数：100 元/点
TARGET_DELTA_ABS = 0.15  # 目标单腿 Delta 绝对值
DAYS_TO_EXPIRY = 25
ROLL_DAYS = 5
SIGMA = 0.22  # 年化波动率假设，实际根据接口查询


def delta(S, K, T, sigma, option_type='call'):
    """
    计算期权的Delta值（基于布莱克-斯科尔斯模型）
    如果有接口可以直接接入，如果没有使用欧式期权定价模型近似
    参数：
    S: 标的资产当前价格（float）
    K: 行权价（float）
    sigma: 波动率（年化，float，如0.2表示20%）
    T: 剩余到期时间（年，float，如0.5表示半年）
    option_type: 期权类型（'call'或'put'，默认'call'）

    返回：
    delta: 期权的Delta值（float）
    """

    if option_type == 'call':
        if T <= 0:
            return 1.0 if S > K else 0.0  # 时间价值归零的时候实值delta为1虚值为0
        d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)  # 看涨期权Delta = N(d1)
    elif option_type == 'put':
        d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1) - 1  # 看跌期权Delta = N(d1) - 1
    else:
        raise ValueError("option_type必须为'call'或'put'")
    print('delta =', round(delta, 4))
    return round(delta, 4)  # 保留4位小数


def find_option_by_delta(option_chain, S, target_delta, option_type, T_days, sigma=0.2):
    """
    在给定到期日的期权链中，找到 Delta 最接近目标值的期权
    :param option_chain: DataFrame，模拟数据，包含字段 ['strike', 'type', 'days_to_expiry']
    :param S: 当前标的资产价格
    :param target_delta: 目标 Delta 绝对值（如 0.15）
    :param option_type: 'call' 或 'put'
    :param T_days: 到期天数（如 25）
    :param sigma: 用于估算 Delta 的波动率
    :return: 最优期权行权价及相关信息
    """
    T = T_days / 252.0  # 将天数转换为年（假设 252 个交易日）

    # 筛选出指定类型和到期日的期权
    options = option_chain[
        (option_chain['type'] == option_type) &
        (option_chain['days_to_expiry'] == T_days)
        ].copy()

    if options.empty:
        raise ValueError(f"在 {T_days} 天到期的期权中未找到 {option_type} 合约")

    # 计算每个行权价对应的 Delta
    deltas = []
    for _, row in options.iterrows():
        K = row['strike']
        if option_type == 'call':
            d = delta(S, K, T, sigma, 'call')
        else:
            d = delta(S, K, T, sigma, 'put')
        deltas.append(d)

    options['delta'] = deltas

    # 确定目标 Delta 值（看涨为正，看跌为负）
    target = -abs(target_delta) if option_type == 'put' else abs(target_delta)

    # 找出 Delta 最接近目标值的合约
    options['delta_diff'] = abs(options['delta'] - target)
    best = options.loc[options['delta_diff'].idxmin()]
    return best

class RiskReversal:
    def __init__(self, target_delta=0.15,days_to_expiry=25, roll_days=5, sigma=0.22, delta_tolerance=0.02):
        # 实际数据根据接口获取为准
        self.target_delta = target_delta
        self.days_to_expiry = days_to_expiry
        self.roll_days = roll_days
        self.sigma = sigma
        self.position = {
            'put': None,   # {'strike', 'qty', 'expiry'}
            'call': None,
            'next_expiry': None
        }
        self.history = []
        self.delta_tolerance = delta_tolerance

    def risk_reversal(self, S, today ):
        """
        使delta为0，保持中性,计算组合总 Delta
        :param S:当前标的资产价格
        :param today:当前日期
        :return:总delta
        """
        total = 0.0
        if self.position['put']:
            T = (self.position['put']['expiry'] - today).days
            if T > 0:
                d = delta(S,self.position['put']['strike'], T / 252.0, self.sigma,option_type='put')
                total += self.position['put']['qty'] * d * INDEX_MULTIPLIER

        if self.position['call']:
            T = (self.position['call']['expiry'] - today).days
            if T > 0:
                d = delta(S,self.position['call']['strike'], T / 252.0, self.sigma, option_type='call')
                total += self.position['call']['qty'] * d * INDEX_MULTIPLIER

        return total

    def open_position(self, S, today, option_chain):
        expiry = today + timedelta(days=self.days_to_expiry)
        chain = option_chain[option_chain['expiry'] == expiry]

        # 初始：买入1张Put，卖出1张Call
        put_opt = find_option_by_delta(chain, S, self.target_delta, 'put', self.days_to_expiry, self.sigma)
        call_opt = find_option_by_delta(chain, S, self.target_delta, 'call', self.days_to_expiry, self.sigma)

        self.position.update({
            'put': {'strike': put_opt['strike'], 'qty': 1, 'expiry': expiry},
            'call': {'strike': call_opt['strike'], 'qty': -1, 'expiry': expiry},
            'next_expiry': expiry
        })

        init_delta = self.risk_reversal(S, today)
        self.history.append({
            'date': today,
            'action': f"【开仓】买入 Put K={put_opt['strike']} x1，卖出 Call K={call_opt['strike']} x1",
            'spot': S,
            'total_delta': init_delta
        })

    def adjust_option_legs(self, S, today, option_chain):
        """
        通过更换行权价或调整头寸数量，使组合 Delta 接近 0
        """
        days_left = (self.position['next_expiry'] - today).days
        if days_left <= self.roll_days:
            return

        chain = option_chain[option_chain['expiry'] == self.position['next_expiry']]
        current_delta = self.risk_reversal(S, today)

        # 若 Delta 超出容忍范围，重新选腿
        tolerance_points = self.delta_tolerance * INDEX_MULTIPLIER  # 例如 0.02 * 100 = 2 点
        if abs(current_delta) <= tolerance_points:
            return

        try:
            # 寻找新的 Put 和 Call，使总 Delta ≈ 0
            # 简化：仍用 ±15-delta，因 Risk Reversal 本身接近中性
            new_put = find_option_by_delta(chain, S, self.target_delta, 'put', days_left, self.sigma)
            new_call = find_option_by_delta(chain, S, self.target_delta, 'call', days_left, self.sigma)

            old_put_k = self.position['put']['strike']
            old_call_k = self.position['call']['strike']

            self.position['put']['strike'] = new_put['strike']
            self.position['call']['strike'] = new_call['strike']

            new_delta = self.get_total_delta(S, today)
            self.history.append({
                'date': today,
                'action': f"【调整腿】Put: {old_put_k}→{new_put['strike']}, Call: {old_call_k}→{new_call['strike']}",
                'spot': S,
                'total_delta_before': current_delta,
                'total_delta_after': new_delta
            })
        except Exception as e:
            print(f"调整失败 ({today}): {e}")

