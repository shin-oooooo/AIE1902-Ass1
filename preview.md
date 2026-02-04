# 说明与投资建议

<table style="width: 100%; border-collapse: collapse;">
  <thead>
    <tr style="background-color: #f8f9fa;">
      <th style="width: 60%; padding: 12px; border-bottom: 2px solid #ddd; text-align: left;">说明与投资建议（面向零基础）</th>
      <th style="width: 40%; padding: 12px; border-bottom: 2px solid #ddd; text-align: left;">名词解释与原理拆解</th>
    </tr>
  </thead>
  <tbody>
    <!-- 1. 数据来源 -->
    <tr>
      <td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;">
        <strong>1. 数据从哪里来、怎么变成可用的“参数”</strong><br><br>
        <strong>(1) 数据获取与清洗</strong><br>
        我们从 AkShare 接口获取了 12 个标的的历史日度收盘价。原始价格数据可能包含录入错误或极端的黑天鹅事件（如暴跌 99%），这会干扰模型判断。因此，我们使用 <strong>Z-Score</strong> 方法识别并剔除统计学意义上的离群点，确保输入数据的纯净性。<br><br>
        <strong>(2) 数据集划分</strong><br>
        为了严谨验证模型能力，我们将时间序列切分为两段：<br>
        - <strong>训练集</strong>：用于让模型“学习”历史规律。<br>
        - <strong>测试集</strong>：用于“考试”，检验模型在未知未来的表现。<br>
        这能有效防止“死记硬背”历史答案（过拟合）。
      </td>
      <td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;">
        <strong>Z-Score (标准分数)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率序列。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算序列均值 μ 与标准差 σ，将每个收益率 x 转化为 z = (x - μ) / σ。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 量化“偏离正常水平的程度”。若 |z|>3，判定为异常并剔除，防止极端值扭曲统计结果。<br><br>
        <strong>训练集 / 测试集 (Train/Test Set)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 完整的时间序列数据。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 按时间点一分为二：前 80% 仅用于计算参数，后 20% 仅用于验证预测。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 模拟真实的“未知未来”场景，确保评估出的模型能力不是靠“偷看答案”得来的。
      </td>
    </tr>

    <!-- 2. 图表分析 -->
    <tr>
      <td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;">
        <strong>2. 图表与统计指标如何影响结论</strong><br><br>
        <strong>(1) 统一比较基准：日收益率</strong><br>
        不同标的价格（100元 vs 3000元）无法直接对比。我们计算 <strong>日收益率</strong>（每日涨跌百分比）作为所有分析的基础。<br><br>
        <strong>(2) 风险与收益的可视化</strong><br>
        - <strong>归一化</strong>走势图帮助我们直观对比谁涨得快、谁更稳。<br>
        - <strong>滚动波动率</strong>（如30日窗口）展示了风险随时间的变化，避免被长期平均值掩盖近期波动。（数据实证：在本次回测周期中，**AU0 (黄金)** 展现了惊人的防御性，年化波动率仅 18%（与大盘 SPY 持平），相比之下，**AVGO** 的波动率高达 58%，意味着持有它的心理压力是持有黄金的三倍以上。）<br>
        - <strong>核密度估计 (KDE)</strong> 描绘了收益率的“概率地图”，让我们看到极端暴涨暴跌出现的可能性。<br><br>
        <strong>(3) 资产间的联动：相关性</strong><br>
        <strong>相关系数</strong> 告诉我们资产是否“同涨同跌”。<strong>相关性热力图</strong> 用颜色直观展示了这一点：深红代表高度同步（风险无法分散），深蓝代表走势相反（具有对冲价值）。（数据洞察：热力图清晰地展示了“科技抱团”现象——NVDA 与 TSMC 的相关性高达 0.62，意味着它们往往齐涨齐跌。唯独 **AU0** 呈现出珍贵的“蓝海”，与 AMZN (-0.11)、MSFT (-0.06) 等巨头均为负相关。这就是为什么算法会重仓黄金——它是唯一能有效对冲科技股崩盘风险的“减震器”。）<br><br>
        <strong>(4) 综合性价比：夏普比率</strong><br>
        我们将 <strong>年化收益率</strong> 与 <strong>年化波动率</strong> 画在同一张图上，并用 <strong>夏普比率</strong> 衡量“性价比”：承担单位风险能换来多少超额收益。（结论：数据告诉我们要“买稳赚的”而不是“买涨得猛的”。虽然 AVGO 涨幅惊人，但其单位风险回报率（夏普 1.25）远低于黄金（2.94）和谷歌（1.96）。优化器会自动惩罚那些“大起大落”的标的。）<br><br>
        <strong>(5) 深度图表解读：为什么这样配置？</strong><br>
        <strong>1) 相关性热力图 (Correlation Heatmap) 给了我们什么信息？</strong><br>
        这张图展示了资产两两之间的联动程度。AU0 (黄金) 所在的行/列呈现出大量的冷色调（蓝色/浅色），相关系数多在 0 附近甚至为负（如与 SPY 的相关性仅为 -0.04）。<br>
        <span style="color: #0056b3; font-weight: bold;">结论：</span>黄金与科技股/美股大盘几乎“绝缘”。当股市剧烈震荡时，黄金往往能走出独立行情。因此，在组合中配置 37.69% 的 AU0 并不是为了搏取最高收益，而是利用这种“不相关性”来充当组合的减震器，大幅降低整体回撤风险。<br><br>
        <strong>2) 夏普比率图 (Sharpe Ratio Plot) 揭示了什么？</strong><br>
        在图中，AU0 位于左上区域（低波动、较高收益），夏普比率高达 2.94，是全场“性价比”之王；而 AVGO 虽然收益极高（年化 73%），但波动率也最大（年化 59%），位于图表最右侧。<br>
        <span style="color: #0056b3; font-weight: bold;">结论：</span>单一押注高收益资产（如 AVGO/NVDA）虽然诱人，但性价比（夏普比率）其实不如稳健资产。优化算法之所以给 AVGO 分配较少权重（1.04%），正是因为它的“单位风险回报”不如黄金划算。投资的真谛不在于“最能涨”，而在于“最稳地涨”。<br><br>
        <strong>3) 滚动波动率 (Rolling Volatility) 说明了什么？</strong><br>
        通过观察 30 日滚动波动率曲线，我们可以看到 NVDA, AVGO 等科技股的曲线经常出现剧烈的尖峰（Spike），说明其风险具有突发性和聚集性；而 SPY 和 MSFT 的曲线则相对平缓。<br>
        <span style="color: #0056b3; font-weight: bold;">结论：</span>科技成长股的持仓体验往往是“过山车”式的。为了平滑这种心理压力，必须引入波动率曲线平缓的基石资产（如 SPY, MSFT, AU0），确保你在市场恐慌时依然拿得住筹码，避免倒在黎明前。
      </td>
      <td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;">
        <strong>日收益率 (Daily Return)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 每日收盘价 P_t。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算 (P_t - P_{t-1}) / P_{t-1}。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 消除价格绝对值差异，将“金额变动”转化为可跨资产比较的“涨跌比例”。<br><br>
        <strong>归一化 (Normalization)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 多只股票的价格序列。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 将每只股票第1天价格设为1.0，后续价格按涨跌幅同比例缩放。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 强制所有曲线从同一起跑线出发，直观对比相对强弱。<br><br>
        <strong>滚动波动率 (Rolling Volatility)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 日收益率序列。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 逐日移动一个 30 天窗口，计算窗口内收益率的标准差。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 捕捉风险的动态变化（如某个月市场突然恐慌），比单一的“平均波动率”更敏感。<br><br>
        <strong>核密度估计 (KDE)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率分布。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 使用高斯核函数对直方图进行平滑拟合。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 绘制出平滑的概率山峰图，直观展示“常态”在哪里，“极端风险”有多大概率。<br><br>
        <strong>相关系数 (Correlation)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 两只股票的收益率序列。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算协方差除以各自标准差的乘积。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 得到 [-1, 1] 的数值：1代表完全同步（无分散效果），-1代表完全对冲（风险抵消）。<br><br>
        <strong>相关性热力图 (Heatmap)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 12x12 相关系数矩阵。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 将数值映射为颜色（红=正相关，蓝=负相关）。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 将枯燥的数字矩阵转化为视觉图谱，一眼识别出哪些资产是“抱团”的。<br><br>
        <strong>年化收益率 (Annualized Return)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 日均收益率。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 日均值 × 252（年交易日）。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 将短期数据扩展为符合直觉的“年回报率”概念。<br><br>
        <strong>年化波动率 (Annualized Volatility)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 日收益率标准差。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 日标准差 × √252。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 统一量纲，便于与年化收益率进行比较。<br><br>
        <strong>夏普比率 (Sharpe Ratio)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 年化收益率与年化波动率。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> (收益 - 无风险利率) / 波动率。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 衡量“性价比”：每承担 1 单位风险，能换来多少超额回报。
      </td>
    </tr>

    <!-- 3. LightGBM -->
    <tr>
      <td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;">
        <strong>3. LightGBM 预测模型如何产生“期望收益”</strong><br><br>
        <strong>(1) 特征工程：从历史中提取规律</strong><br>
        模型不仅看昨天的涨跌，还通过 <strong>特征 (Features)</strong> 观察更多维度：<br>
        - <strong>滞后项</strong>：过去第1天、第2天...的收益。<br>
        - <strong>动量</strong>：过去一段时间的累计涨幅趋势。<br>
        - 滚动统计：近期的平均水平和波动状态。<br><br>
        <strong>(2) 预测引擎：LightGBM</strong><br>
        相比于简单的“历史均值”（Naive），<strong>LightGBM</strong> 是一种强大的机器学习算法，能捕捉复杂的非线性规律。它在 <strong>测试集</strong> 上的表现通过以下指标衡量：<br>
        - <strong>MAE / RMSE</strong>：预测数值的误差大小（越小越好）。<br>
        - <strong>方向准确率</strong>：猜对“涨/跌”方向的概率（越高越好）。<br><br>
        最终，模型输出每个标的未来的“预期日均收益率”，构成我们的投资导航图。
      </td>
      <td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;">
        <strong>特征 (Features)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 原始价格/收益序列。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算统计量（如5日均线、10日波动等）作为新变量。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 将单一的价格信息扩展为多维度的“市场状态描述”，供模型学习。<br><br>
        <strong>滞后项 (Lag)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 将时间轴平移（如用昨天的 t-1 预测今天的 t）。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 捕捉“惯性”或“反转”效应（如昨日大跌今日往往反弹）。<br><br>
        <strong>动量 (Momentum)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 过去 N 天的收益率。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算期间累计涨跌幅。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 量化趋势强度，判断当前是处于上升期还是下跌期。<br><br>
        <strong>LightGBM</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 构造好的特征矩阵。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 训练数百棵决策树，每棵树修正前者的残差，最终集成投票。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 相比线性模型，能捕捉复杂的非线性市场规律（如“跌多了且波动大时容易反弹”）。<br><br>
        <strong>MAE / RMSE</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 预测值 vs 真实值。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算两者差值的绝对值平均或平方根。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 数值越小，说明模型预测的“点位”越精准。<br><br>
        <strong>方向准确率</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 预测正负号 vs 真实正负号。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 统计符号一致的比例。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 即使点位不准，若能猜对“涨跌方向”，对投资依然极具价值。
      </td>
    </tr>

    <!-- 4. 优化 -->
    <tr>
      <td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;">
        <strong>4. 均值-方差优化如何变成最终权重</strong><br><br>
        <strong>(1) 两大核心输入</strong><br>
        要计算最佳投资比例，我们需要两个数学对象：<br>
        - <strong>期望收益向量 ($\mu$)</strong>：由 LightGBM 预测的 12 个标的未来日均收益。<br>
        - <strong>协方差矩阵 ($\Sigma$)</strong>：描述 12 个标的之间波动大小和联动关系的矩阵。<br><br>
        <strong>(2) 寻找最优解：蒙特卡洛模拟</strong><br>
        我们利用 <strong>均值-方差优化</strong> 理论，通过 <strong>蒙特卡洛</strong> 方法随机生成数十万种投资组合。在“中风险偏好”下，我们寻找那个既能提供不错收益，又能通过分散配置将波动控制在合理范围的组合。<br><br>
        <strong>(3) 最终产出：权重</strong><br>
        算法最终给出一组 <strong>权重</strong>（百分比），告诉我们每 100 元资金应分别投向哪些资产。
      </td>
      <td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;">
        <strong>期望收益向量 (μ)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> LightGBM 模型的预测输出。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 整理为 12x1 列向量。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 告诉优化器每个资产“未来可能赚多少”。<br><br>
        <strong>协方差矩阵 (Σ)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率序列。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 计算两两之间的协方差，构成 12x12 矩阵。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 告诉优化器“哪些资产一起动”，从而指导分散配置。<br><br>
        <strong>均值-方差优化</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 输入 μ 和 Σ。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 求解数学规划问题：在风险约束下最大化收益。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 找到理论上的“最佳性价比”配方。<br><br>
        <strong>蒙特卡洛 (Monte Carlo)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 随机数生成器。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 随机尝试 20 万种不同的权重组合，计算每一组的结果。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 暴力穷举出近似的最优解，解决复杂数学方程难以直接求解的问题。<br><br>
        <strong>权重 (Weights)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 优化算法的最优解。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 归一化使得总和为 100%。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 直接转化为可执行的资金分配指令（如“买 30% 黄金”）。
      </td>
    </tr>

    <!-- 5. 投资建议 -->
    <tr>
      <td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;">
        <strong>5. 你应该如何进行最后投资</strong><br><br>
        基于“中风险偏好”的计算结果，我们建议的资金分配方案如下：<br><br>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9em; border: 1px solid #eee;">
            <tr style="background-color: #f0f2f6;">
                <th style="text-align: left; padding: 4px;">标的</th>
                <th style="text-align: right; padding: 4px;">权重</th>
                <th style="text-align: left; padding: 4px;">角色定位</th>
            </tr>
            <tr><td>AU0</td><td style="text-align: right;">37.69%</td><td><strong>防守核心</strong>：黄金，对冲股市波动</td></tr>
            <tr><td>MSFT</td><td style="text-align: right;">25.39%</td><td><strong>稳健成长</strong>：科技巨头，基石仓位</td></tr>
            <tr><td>SPY</td><td style="text-align: right;">12.05%</td><td><strong>市场基准</strong>：标普500 ETF，分散风险</td></tr>
            <tr><td>AMZN</td><td style="text-align: right;">7.83%</td><td><strong>进攻</strong>：电商与云服务龙头</td></tr>
            <tr><td>AAPL</td><td style="text-align: right;">3.78%</td><td><strong>进攻</strong>：消费电子龙头</td></tr>
            <tr><td>TSMC</td><td style="text-align: right;">3.74%</td><td><strong>进攻</strong>：半导体制造核心</td></tr>
            <tr><td>META</td><td style="text-align: right;">3.17%</td><td><strong>进攻</strong>：社交网络与元宇宙</td></tr>
            <tr><td>GOOGL</td><td style="text-align: right;">2.41%</td><td><strong>进攻</strong>：搜索与广告业务</td></tr>
            <tr><td>ORCL</td><td style="text-align: right;">1.72%</td><td><strong>进攻</strong>：企业软件服务</td></tr>
            <tr><td>AVGO</td><td style="text-align: right;">1.04%</td><td><strong>进攻</strong>：半导体设计</td></tr>
            <tr><td>NVDA</td><td style="text-align: right;">0.64%</td><td><strong>高波进攻</strong>：AI 算力核心（控风险）</td></tr>
            <tr><td>ASML</td><td style="text-align: right;">0.55%</td><td><strong>进攻</strong>：光刻机垄断</td></tr>
        </table><br>
        <strong>操作指南：</strong><br>
        1. <strong>长期定投（推荐）</strong>：按上述比例首次建仓后，每月投入固定资金，依然按此比例分配。定期（如每季度）检查持仓，若某资产占比偏离超过 5%，则卖高买低进行<strong>再平衡</strong>。<br>
        2. <strong>风险控制</strong>：严格遵守纪律，不因短期涨跌随意更改配方。若必须做短线，请设置严格止损线（如 -5%）。（策略逻辑：为何黄金占比高达 37%？这不是巧合，而是数学上的必然。因为黄金是组合中唯一的“负相关资产”，为了让整体曲线平滑，算法必须用足够多的黄金来中和科技股的剧烈波动。）
      </td>
      <td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;">
        <strong>再平衡 (Rebalance)</strong><br>
        <span style="color: #444; font-weight: bold;">[数据]</span> 账户当前持仓 vs 目标权重。<br>
        <span style="color: #444; font-weight: bold;">[处理]</span> 卖出涨幅过大导致占比超标的资产，买入占比不足的资产。<br>
        <span style="color: #444; font-weight: bold;">[效果]</span> 强制实现“高抛低吸”，维持组合的风险特征不随市场波动而漂移。
      </td>
    </tr>
  </tbody>
</table>
