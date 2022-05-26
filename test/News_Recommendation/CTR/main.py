import time
import torch
import numpy as np
import random

from torchnlp.word_to_vector import GloVe
from torchtext.vocab import Vectors

from utils import parse_args, MINDDataset, HeyDataset
from mylog import Logger
from nrms_models import NRMS

'''
运行环境:
    python=3.6.13
	requirements.txt
核心代码思想: 
	(1)数据加载: 一个impression表示一条行为记录, 一个click表示一条行为记录中一个候选正样本; 新闻ctr值的统计: 点击量/曝光量, 前者统计每一条记录中的候选正样本, 后者为上述的impressions值
		MIND(small): [数据链接](https://msnews.github.io/index.html)
			news.tsv: 以'\t'隔开, 第一个为newsid, 第二个为category one, 第三个为category two, 第四个为title(word list)
            	[得到一个news字典, key为newsid, value为一个元组, 包含category one, category two和title的word list]
				[之后再得到一个newsid2index的字典newsidenx]
				[针对newsidenx字典中的index顺序, 可分别得到word_dict, categ_dict和news_features列表(以title和category作为news feature, 其中title被title_size限制)]
			behaviors.tsv: 以'\t'隔开, 其中第二个为用户id, 第四个为news id序列(历史点击序列), 第五个为候选news序列
				[每一个正样本匹配若干个负样本，得到一个多分类问题]
            	[历史点击序列被his_size限制, 训练集中针对每一个用户分别统计正负候选news, 然后一个训练样例由随机选取negnums个负候选news和一个正候选news构成的列表, 样例对应的标签为正候选news的索引, 此外该训练样例还包括用户历史点击序列和实际的点击序列长度(用于mask)]
            	[对于测试集, 历史点击序列同样被his_size限制, 针对每一个用户, 统计所有的正负候选news及其1/0标签, 每一个用户对应一个测试样例, 包括该用户的历史点序列/实际点击序列长度/正负候选news列表/对应的1/0标签列表]
		heybox: 原始数据包括用户日志数据(形如xxxx-xx-xx-contact.csv)、帖子数据(形如links_x.json); 以MIND的方式处理数据
			news.tsv: 以'\t'隔开, 第一个为帖子id, 第二个为用户id, 第三个为主题, 第四个为标题, 第五个为正文
            	[得到一个news字典, key为帖子id, value为一个元组, 包含用户id, 主题和标题的word list]
				[之后再得到一个帖子id2index的字典newsidenx]
				[针对newsidenx字典中的index顺序, 可分别得到word_dict, post_user_dict, categ_dict和news_features列表(以标题, 用户id, 主题作为post feature, 其中标题被title_size限制)]
        	behaviors.tsv: 以'\t'隔开, 第一个为索引值, 第二个为用户id, 第三个为时间戳, 第四个为历史点击序列, 第五个为候选样本序列(包括多个负样本和一个正样本)
            	[历史点击序列被his_size限制, 训练集中针对每一个用户分别统计正负候选帖子, 然后一个训练样例由随机选取negnums个负候选帖子和一个正候选帖子构成的列表, 样例对应的标签为正候选帖子的索引, 此外该训练样例还包括用户历史点击序列和实际的点击序列长度(用于mask)]
            	[对于测试集, 历史点击序列同样被his_size限制, 针对每一个用户, 统计所有的正负候选样本及其1/0标签, 每一个用户对应一个测试样例, 包括该用户的历史点序列/实际点击序列长度/正负候选帖子列表/对应的1/0标签列表]
	(2)加载预训练嵌入:
		英文数据集对应GloVe840B
		中文数据集对应sgns.zhihu.word
	(3)模型: 均使用候选新闻ctr(优化匹配分数)
		单用户特征模型:
			NRMS: 分别获得候选新闻表示和用户表示, 其中新闻表示包括标题表示和主题类表示, 标题表示使用多头注意力和加性注意力; 用户表示在新闻表示的基础上也使用多头注意力和加性注意力
			NRMS+din: 使用目标注意力替换自注意力
			NRMS+ctr: 用户新闻序列ctr(优化用户表示)
		混合特征模型:
			NRMS+scalar: 将每个单用户特征的匹配分数相加
			NRMS+pool: 对用户特征进行池化操作, 包括加法, 均值, 最大以及加权
			NRMS+concat: 将用户特征进行连接操作, 然后作为DNN的输入, DNN可考虑MOE或者MVKE或者cross_gate
	(4)训练和测试
		指标: 
			AUC
			MRR: 按照预测概率从大到小排序ground truth, 每个位置除以(索引+1), 最后求和取平均
			NDCG(5/10): 按照预测概率从大到小排序ground truth, 只取前k个, 每个位置除以(索引取对数), 最后求和


MIND:
	[python3 main.py --dataset MIND --pretrained_embeddings glove]
		auc:0.663, mrr:0.317, ndcg5:0.349, ndcg10:0.411
	[python3 main.py --dataset MIND --pretrained_embeddings glove --din]
		auc:0.663, mrr:0.316, ndcg5:0.347, ndcg10:0.409
	[python3 main.py --dataset MIND --pretrained_embeddings glove --use_ctr]
		auc:0.663, mrr:0.317, ndcg5:0.348, ndcg10:0.411

	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --score_add]
		auc:0.665, mrr:0.319, ndcg5:0.350, ndcg10:0.413
	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --add_op]
		auc:0.666, mrr:0.321, ndcg5:0.352, ndcg10:0.415
	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --mean_op]
		auc:0.666, mrr:0.320, ndcg5:0.351, ndcg10:0.414
	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --max_op]
		auc:0.665, mrr:0.317, ndcg5:0.349, ndcg10:0.412
	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --atten_op]
		auc:0.665, mrr:0.320, ndcg5:0.351, ndcg10:0.414

	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --dnn]
		auc:0.660, mrr:0.315, ndcg5:0.345, ndcg10:0.408
	[python3 main.py --dataset MIND --pretrained_embeddings glove --din --use_ctr --moe]
		auc:0.660, mrr:0.314, ndcg5:0.343, ndcg10:0.408

heybox:
	[python3 main.py --dataset heybox --pretrained_embeddings sgns --title_size 10 --batch_size 512 --neg_number 10 --lr 1e-5 --epochs 5]
		auc:0.657, mrr:0.458, ndcg5:0.520, ndcg10:0.587
	[python3 main.py --dataset heybox --pretrained_embeddings sgns --title_size 10 --batch_size 512 --neg_number 10 --lr 1e-5 --epochs 5 --din]
		auc:0.660, mrr:0.462, ndcg5:0.523, ndcg10:0.590
	[python3 main.py --dataset heybox --pretrained_embeddings sgns --title_size 10 --batch_size 512 --neg_number 10 --lr 1e-5 --epochs 5 --use_ctr]
		auc:0.657, mrr:0.458, ndcg5:0.520, ndcg10:0.587

	[python3 main.py --dataset heybox --pretrained_embeddings sgns --title_size 10 --batch_size 512 --neg_number 10 --lr 1e-5 --epochs 5 --din --use_ctr --score_add]
		auc:0.661, mrr:0.464, ndcg5:0.525, ndcg10:0.592
	[python3 main.py --dataset heybox --pretrained_embeddings sgns --title_size 10 --batch_size 512 --neg_number 10 --lr 1e-5 --epochs 5 --din --use_ctr --add_op]
		auc:0.659, mrr:0.461, ndcg5:0.523, ndcg10:0.590
	[python3 main.py --dataset heybox --pretrained_embeddings sgns --title_size 10 --batch_size 512 --neg_number 10 --lr 1e-5 --epochs 5 --din --use_ctr --moe]
		auc:0.659, mrr:0.460, ndcg5:0.522, ndcg10:0.589

数据集信息:
|  | MIND | heybox |
|---|---|---|
| # users | 94057 | 230343 |
| # news/posts | 65238 | 737507 |
| # ave words in news/post title | 11.77 | 5.91 |
| # impressions | 230117 | 5332771 |
| # clicks | 347727 | 5332771 | 
| # train samples | 236344 | 2919193 |
| # test samples | 73152 | 2413578 | 
'''

args = parse_args()
savepath = 'result/' + args.modelname + '/' + args.dataset + '/' + time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
log = Logger('root', savepath)
args.savepath = savepath
logger = log.getlog()
write_para = ''
for k, v in vars(args).items():
	write_para += '\n' + k + ' : ' + str(v)
logger.info('\n' + write_para + '\n')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

if args.dataset == 'MIND':
	data = MINDDataset(args)
elif args.dataset == 'heybox':
	data = HeyDataset(args)

if args.pretrained_embeddings == 'glove':
	preemb = GloVe(name='840B', cache="pretrained/glove")
elif args.pretrained_embeddings == 'sgns':
	preemb = Vectors(name='sgns.zhihu.word', cache='pretrained/sgns')
else:
	preemb = None
	
if args.modelname == 'nrms':
	model = NRMS(preemb,args,logger,data)
model.mtrain()