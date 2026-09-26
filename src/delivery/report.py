"""Render the combined competition method report, with embedded Chinese font."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def build_report(destination):
    candidates = [Path('C:/Windows/Fonts/simhei.ttf'), Path('/usr/share/fonts/truetype/arphic/uming.ttc')]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path:
        pdfmetrics.registerFont(TTFont('CN', str(font_path)))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    font = 'CN' if font_path else 'STSong-Light'
    styles = {
        'title': ParagraphStyle('title', fontName=font, fontSize=23, leading=32, textColor=colors.HexColor('#16344B'), spaceAfter=20),
        'h': ParagraphStyle('h', fontName=font, fontSize=15, leading=22, spaceAfter=12, textColor=colors.HexColor('#16344B')),
        'p': ParagraphStyle('p', fontName=font, fontSize=10, leading=16, spaceAfter=10, wordWrap='CJK', alignment=TA_LEFT),
        'small': ParagraphStyle('small', fontName=font, fontSize=8.5, leading=13, spaceAfter=8, wordWrap='CJK'),
    }
    story = []
    def p(text, kind='p'):
        story.append(Paragraph(escape(str(text)).replace('\n', '<br/>'), styles[kind]))
    def table(rows, widths):
        cells = [[Paragraph(escape(str(c)), styles['small']) for c in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6EFF4')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#A6BDCD')),
            ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#D5DFE5'))]))
        story.extend([t, Spacer(1, 12)])
    def page(title):
        story.append(PageBreak())
        p(title, 'h')
    b = read('outputs/prediction/evaluation.json')
    c = read('outputs/scoring/evaluation.json')
    frozen = read('outputs/scoring/frozen_scoring.json')
    robust = read('outputs/prediction/robustness.json')
    p('驾驶安全\n事故预测与司机安全评价方案', 'title')
    p('Crazy Logi-AI 2.0 · 算法与模型说明', 'h')
    p('方案概述', 'h')
    p('本方案面向物流车队的事故预防和司机安全管理，使用组委会提供的车辆画像、风险事件、行驶轨迹及IMU数据，为500辆车输出未来40天事故风险预测，并建立可解释、可持续更新的司机安全评分体系。')
    p('事故预测采用正式目标与辅助风险目标的浅层LightGBM模型，通过训练期暴露量收缩、三随机种子平均和等权排名融合学习风险排序，再进行概率校准与预测期限换算。安全评分以历史高危事件、疲劳分心、驾驶稳定性、道路规则、管理配合和IMU佐证六个维度为基础，结合时间衰减、暴露校正及固定参考分布形成0至100分。')
    table([['任务', '输出与用途'],
           ['事故预测', '每辆车输出二分类accident及连续概率risk_prob；按风险排序安排优先复核'],
           ['安全评价', '每名司机输出安全分、排名、等级、分项贡献、前三项原因和管理建议'],
           ['落地与复现', '提供数据处理、特征计算、训练预测、动态评分源码与安装运行说明']], [100, 400])
    p('样本与目标定义', 'h')
    p('数据实际覆盖2026年6月1日至7月31日，共61天；gpsno作为车辆与司机的一一对应标识。正式正类定义为目标窗口内事故事件11803至少1条，或未遂事件11804至少2条。计数单位为事件记录，单条未遂可作为历史行为证据，但不单独构成正式正类。')
    p('最终预测截止日为2026年7月31日，预测区间为2026年8月1日至9月9日，共40天，计算采用左闭右开区间[2026-08-01,2026-09-10)。只使用比赛提供的数据，不引入外部训练数据。')
    p('总体流程：原始数据清洗 → 车辆日级汇总与时间标签 → 固定窗口特征 → 事故预测 / 六维安全评分 → 逐车结果与管理建议。历史评估用于检验风险区分能力；40天绝对概率的适用条件在预测方法与验证章节说明。', 'small')

    page('01 | 数据处理与特征工程')
    table([['数据来源', '规模', '处理结果'],
           ['车辆画像', '500辆', '字符串ID、唯一键和车辆集合校验'],
           ['风险事件', '3,152,654条', '逐事件表及完整车辆日期日历'],
           ['行驶轨迹', '584,310,388条', '里程、工时、速度、覆盖与质量日表'],
           ['车载IMU', '738,244,484条', '各轴、模长、异常比例及缺失日表']], [85, 120, 295])
    p('清洗规则：使用DuckDB流式读取gzip并输出压缩Parquet。时间限制为[2026-06-01,2026-08-01)，事件类型按编码识别。有效速度为0至160km/h；异常速度、坐标保留质量标记。距离由厘米转公里，负距离和负时长不进入暴露汇总。风险记录不会因位置异常而丢失标签，IMU缺失保持未知，不补造采样记录。')
    p('候选特征覆盖事件次数、暴露归一风险率、事件日密度、单日聚集性、近期趋势、历史事故与未遂、轨迹速度分布、IMU统计及传感器覆盖率。共构造795个候选特征，仅在最早训练窗口删除全空、常量和高度相关列（绝对相关系数≥0.995），按特征族轮转保留100个核心列。')
    p('窗口一致性：核心列中93列使用固定14天观察窗，7列使用最近7天统计；最终预测与历史训练使用同一窗口含义。17个时间窗口共8,500行特征样本，用于不同期限的构造与审计；预测模型只使用14天观察、14天目标的样本。重复窗口来自同一批500辆车，不视为独立车辆。')
    p('辅助模型使用100个核心列和7个收缩风险率，覆盖历史高危、疲劳、分心、稳定性、道路、管理及盲区。对计数n、暴露e，定义 r=(n+τμ)/(e+τ)。μ为训练期事件总量/暴露总量，τ为训练期正暴露中位数。疲劳与分心按工时，其余按公里；零或未知暴露保留缺失。收缩用于降低低暴露下比率估计的波动。')
    p('防止信息泄漏：各窗口只使用截止日以前的信息，训练标签结束日不晚于预测截止日；收缩先验、裁剪阈值、填补值均只在训练期拟合。ID、日期、窗口元数据和未来目标不入模。半年画像统计截至7月31日，因此不作为早期回测的预测特征。')

    page('02 | 事故预测模型与40天概率')
    p('双目标建模：正式模型预测未来14天是否满足“事故≥1或未遂≥2”；辅助模型预测未来14天是否出现任一次事故或未遂。辅助目标提供更稠密的监督信号，两者使用同一时间边界，并统一按正式目标评价。模型采用浅层树与正则约束，以适应车辆数少、正式正类稀疏的特点。')
    table([['参数', '正式目标模型', '辅助目标模型'],
           ['特征数', '100', '107'], ['树数 / 最大深度 / 叶子上限', '63 / 2 / 7', '100 / 3 / 7'],
           ['学习率 / 最小叶样本', '0.03 / 40', '0.03 / 30'], ['L1 / L2正则', '0.5 / 5', '0.5 / 10'],
           ['列采样 / 行采样', '0.7 / 0.8', '0.8 / 0.8'],
           ['类别权重', '类别频数反比', '正类乘以负正样本比平方根'],
           ['随机种子', '2026', '2026、42、137']], [180, 140, 180])
    p('正式模型先按训练期0.5%与99.5%分位裁剪，再用训练中位数填补并附加缺失指示；辅助模型使用LightGBM原生缺失分支。辅助训练按车辆出现窗口数的倒数归一基础权重，再叠加类别权重，减少重复窗口数量不同造成的偏置。')
    p('融合：三个辅助模型的概率先求平均；正式与辅助输出各自转换为完整车队的百分位q1、q2，风险分 R=0.5q1+0.5q2，同分采用平均名次。整批500辆车一起计算；推理检查唯一ID、完整车辆集合与统一截止日。最终训练使用截至7月31日已获得完整标签的14天样本。')
    p('概率校准：在6月30日截止的开发窗口拟合C=1的正则化Platt映射，得到14天概率p14。二分类阈值在开发数据上满足召回率至少70%的候选中最大化“召回率-假阳性率”，随后随预测期限作同一单调换算。')
    p('期限换算：假定事故和未遂是独立、平稳的Poisson计数，日强度分别为λa、λn，则 Pt=1-exp[-t(λa+λn)]×(1+tλn)。用p14及历史事故/未遂条数占比求解强度，再取t=40得到risk_prob；accident为risk_prob达到换算阈值时的0/1标记。该外推保持排序，但40天绝对概率仍需未来真值验证。')

    page('03 | 时间验证与预测效果')
    table([['环节', '特征截止日', '标签期与用途'],
           ['最早训练窗口', '6月14日', '6月15日至28日；特征筛选及初始监督样本'],
           ['开发评估', '6月28日 / 6月30日', '后续14天；参数确定与校准，训练标签须在截止日前可得'],
           ['历史回顾', '7月17日', '7月18日至31日；500辆车、28个正式正类']], [110, 130, 260])
    p('按时间先后划分样本，禁止随机拆分窗口行。每次评价只用当时可获得的训练标签。两个开发窗口标签重叠且各仅6个正类，不能视为独立重复实验。以下晚期结果属于已查看历史窗口的回顾性评估，不作为新的独立测试结论。')
    table([['评价指标', '结果', '含义'],
           ['事故预测ROC-AUC', f"{b['candidate_auc']:.4f}", '衡量正负车辆总体风险排序；比赛主要评价指标'],
           ['事故预测AP', f"{b['candidate_ap']:.4f}", 'Average Precision，衡量稀疏正类的检索表现'],
           ['安全评分风险AUC', f"{c['auc']:.4f}", '使用100减安全分作为风险方向']], [180, 80, 240])
    table([['按预测风险优先复核', '捕获正类', '召回率'],
           *[[f'前{k}辆', robust['retrospective_topk'][str(k)]['positives'],
              f"{robust['retrospective_topk'][str(k)]['recall']:.1%}"] for k in [25, 50, 100]]], [200, 150, 150])
    p('高风险最前段仍存在不稳定性，前25辆仅捕获1个正类。管理使用时应结合评分明细与事件证据确定复核对象，不能把较高总体AUC理解为最前端车辆一定发生事故。')
    for name, label in [('imu_dropout_20%', '随机20%车辆IMU输入缺失'), ('imu_dropout_100%', '全部车辆IMU输入缺失')]:
        r = robust['stress'][name]
        p(f"敏感性检验：{label}时，与完整输入预测的排序Spearman为{r['spearman']:.4f}，前50辆重合率为{r['top50_overlap']:.0%}。", 'small')
    p('三个辅助随机种子间排序Spearman约0.975至0.979。上述检验描述输入或随机种子扰动下的稳定性，不代表未来性能保证。开发正类少、同车窗口重叠和传感器覆盖随时间变化均限制结论外推；当前没有8月以后真值，真实40天风险区分与概率校准尚未验证。', 'small')

    page('04 | 安全评分公式与维度权重')
    p('安全评分采用截至评分日的固定14天观察窗，分数越高表示观测行为风险越低。评分与事故概率独立呈现：前者解释近期行为及管理重点，后者估计目标期事件风险，两者不能直接互换。')
    p('第一步，时间衰减。事件距截止日a天时权重为2^(-a/14)，维度事件强度 N=Σ[事件严重度×2^(-a/14)]。里程、工时和活跃日按相同14天半衰期衰减，得到暴露E。事件按发生时刻计算，日级暴露按当日中点近似。')
    p('第二步，暴露收缩。五个行为维度计算 r=(N+τμ)/(E+τ)。参考μ为训练期总事件强度/总暴露，τ为训练期正暴露中位数。历史、稳定性、道路按每千公里，疲劳分心按每百小时，管理按实际移动车辆日。暴露不足时向参考总体收缩，避免将少行驶或少上报直接视作安全。')
    p('第三步，参考映射。用6月1日至14日的固定参考分布，将风险率映射为惩罚指数P：P=上限×[参考值小于r的数量+0.5×等于r的数量]/参考样本数。五个行为维度上限100，IMU上限60。固定参考保证不同日期的分数可比较。')
    names = {'history': '历史事故与未遂', 'fatigue_distraction': '疲劳与分心', 'driving_stability': '驾驶稳定性',
             'road_rule': '道路规则', 'management': '管理配合', 'imu_evidence': 'IMU佐证'}
    units = {'history': '每千公里', 'fatigue_distraction': '每百小时', 'driving_stability': '每千公里',
             'road_rule': '每千公里', 'management': '每活跃日', 'imu_evidence': '采样记录及日级统计'}
    table([['维度', '权重', '暴露口径'], *[[names[d], f'{w:.2%}', units[d]] for d,w in frozen['weights'].items()]], [180, 100, 220])
    p('第四步，加权合成。维度贡献D=w×P，安全分S=100-ΣD，范围为0至100。贡献包含经验先验收缩，解释中同时披露实际记录及收缩成分。IMU缺失时行为贡献为0，另列U=30×wIMU及保守下界max(0,S-U)，表示证据不充分，不把缺失直接记为违规。')
    p('权重以正式与辅助目标的成对逻辑排序损失等权训练，加入向六维业务先验（35%、20%、20%、10%、10%、5%）的L2收缩（系数2），由SLSQP求解。约束权重非负且和为1、与先验单维偏差不超过0.10、历史权重最高、IMU不超过10%、疲劳及稳定性不低于道路规则。训练使用6月14日评分和后续14天标签，参考分布及权重在动态应用中固定。', 'small')

    page('05 | 行为细则、IMU佐证与可信度')
    p('事件严重度表示相对危险程度，并非直接扣分。每条事件只归属一个行为维度，经过时间衰减、暴露校正和参考映射后再计算贡献；24类事件均纳入规则。')
    table([['维度', '事件类型与严重度'],
           ['历史事故与未遂', '事故10；未遂事故7'],
           ['疲劳与分心', '疲劳（闭眼）5；困倦4.5；打哈欠3.5；看手机3.5；注意力分散3；打电话3；频繁低头3；抽烟2'],
           ['驾驶稳定性', '前碰撞预警5；车距过近3；长时间压线2.5；左/右车道偏移各1.5；后/右盲区报警各2'],
           ['道路规则', '路口超速2.5；主路弯道超速3；危险路段超速3；匝道超速3；高速长下坡超速4'],
           ['管理配合', '摄像头遮挡2；摄像头角度扭转1.5']], [110, 390])
    p('IMU佐证：不假定设备坐标轴对应车体方向，使用加速度模长和角速度模长。令fa为加速度模长不在0.5至1.5范围的记录比例，fg为角速度模长大于20的记录比例，dv为EMS与GPS速度差的非空日均值。速度差可得时 rIMU=0.60fa+0.30fg+0.10×min(dv/20,1)；不可得时去掉该项并将其余项除以0.90。')
    p('IMU指标是风险佐证，不作为事故标签。异常采样点数可能来自同一次持续动作，解释中不将其称为独立危险事件次数。没有IMU记录时，该维度为未知，并通过可信度与不确定性下界披露。')
    p('数据可信度Q由覆盖率、有效率和暴露充分度组成，范围0至100：Q=25×轨迹日覆盖率+15×IMU日覆盖率+15×速度有效率+15×坐标有效率+10×min(衰减里程/1000,1)+10×min(衰减工时/100,1)+10。日覆盖率按14天计算并截断到[0,1]；最后10分表示事件数据源可读取，不代表所有风险事件均被捕获。')
    p('Q≥80为高可信度，60≤Q<80为中可信度，Q<60为低可信度。低可信度优先排查设备与数据覆盖，不直接据分数奖惩。每名司机展示贡献最大的前三个维度，包括记录数量、惩罚指数、扣分贡献及对应建议，明细贡献之和应等于100减安全分。')
    p('权重训练只有4个正式正类，学习到的权重仍受小样本影响。模型通过业务先验、非负约束、IMU扣分上限和人工复核流程约束应用范围。', 'small')

    page('06 | 风险分级与动态运营')
    p('按安全分降序形成排名，名次越小越安全；同分按gpsno确定稳定顺序。A/B/C/D采用20%/40%/30%/10%的固定容量，即500辆车分别为100/200/150/50辆。分级服务于有限管理资源的分配，不代表绝对事故概率；车队整体改善应观察固定参考安全分及行为率，而非等级人数。')
    table([['等级', '车辆数', '后续正类', '观察率 / Wilson 95%区间'],
           *[[k, v['n'], v['positive'], f"{v['rate']:.1%} / {v['wilson_ci95'][0]:.1%}-{v['wilson_ci95'][1]:.1%}"]
             for k,v in c['level_rates'].items()]], [70, 80, 90, 260])
    p('以上为7月17日评分对应7月18日至31日目标的回顾结果。各等级观察率为1%、6%、6%、12%，但置信区间明显重叠，尚不足以证明等级间差异稳定。Wilson区间只反映当前样本的二项抽样不确定性，不能覆盖时间漂移与模型选择偏差。')
    table([['等级 / 状态', '行动与周期'],
           ['A：常规观察', '维持监测；复核数据与事件后实施正向激励'],
           ['B：行为复盘', '每月复盘前三项贡献及可改善行为'],
           ['C：定向培训', '针对主要维度安排培训，每周检查事件与风险率'],
           ['D：优先干预', '优先复核具体事件，形成培训、整改、复查闭环'],
           ['低可信度', '设备、安装角度和上报覆盖检查优先；暂不自动奖惩']], [130, 370])
    p('更新机制：每天、每周或每月触发相同评分流程，始终取截至当日的14天滚动观察窗，使用固定先验、参考分布及权重。新行为进入窗口，旧行为衰减或移出，分数随之变化；执行频率不改变观察窗口长度。')
    p('提交结果截至7月31日，并提供相对7月24日的7日分数变化；7月24日至31日连续8天评分用于验证动态计算。管理者同时关注安全分、风险率、扣分原因与可信度，司机查看自身分数、排名和改进重点。评分配置所需标签截至7月14日，动态应用不得早于该日期；超过已有数据覆盖日需先接入新数据。')

    page('07 | 运行复现与提交文件')
    p('运行环境：Python 3.13，CPU执行，无需GPU。主要依赖DuckDB、pandas、NumPy、LightGBM、scikit-learn和SciPy；PDF生成使用ReportLab。代码包提供锁定依赖、安装运行说明以及数据预处理、特征、训练、预测、评分和校验源码。')
    p('输入目录：\ndata/raw/vehicle_profile.xlsx\ndata/raw/risk_events.tsv.gz\ndata/raw/trajectory/*.gz（30个分片）\ndata/raw/imu/*.gz（30个分片）', 'small')
    p('安装与运行：\npython -m venv .venv\n.venv/Scripts/python.exe -m pip install -r requirements-lock.txt\n.venv/Scripts/python.exe -m src.reproduce_submission --clean --rebuild-features --package', 'small')
    p('已有清洗表和特征：\n.venv/Scripts/python.exe -m src.reproduce_submission --package\n按日期更新评分：\n.venv/Scripts/python.exe -m src.scoring.enhanced --as-of 2026-07-31\n提交文件校验：\n.venv/Scripts/python.exe -m src.delivery.validate submission/final', 'small')
    p('Linux/macOS将解释器路径替换为.venv/bin/python。预测读取固定模型配置，重新拟合各训练期先验、校准器和最终模型；评分读取固定权重及参考分布。全量数据处理采用流式读取，运行时间取决于硬件与磁盘。')
    table([['提交文件', '内容及格式'],
           ['forecast_result.csv', '500行；gpsno唯一；accident为0/1整数；risk_prob为有限[0,1]概率'],
           ['safety_scores.csv', '同一500名司机；安全分[0,100]、排名、等级、六维贡献、可信度、前三项原因、7日变化及建议'],
           ['method_report.pdf', '数据处理、特征、预测、评分、验证与运营方案说明'],
           ['source_code.zip', '源代码、模型配置、依赖与安装运行说明；比赛数据由授权使用者另行提供']], [160, 340])
    p('结果CSV使用UTF-8无BOM与LF换行。验收覆盖车辆集合、预测取值、评分贡献对账、排名与等级一致性。源码包从基础清洗表独立重建特征、训练及评分，已验证两份结果CSV逐字节一致；该复现范围不包含再次扫描全部原始传感器文件。')

    destination.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor('#647C8C'))
        canvas.drawString(47, 28, '驾驶安全 | 事故预测与司机安全评价')
        canvas.drawRightString(A4[0] - 47, 28, str(doc.page))
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=47, leftMargin=47,
                            topMargin=45, bottomMargin=48, title='驾驶安全事故预测与司机安全评价', author='参赛团队')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination
