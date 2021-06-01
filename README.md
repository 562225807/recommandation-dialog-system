# recommandation-dialog-system
## 
```run_language_modeling.py```模型训练
```run_generation.py```生成
# LIC2020 比赛设置
### 预计最大传输长度8192
# useage
```python
python conversation_client.py
```
1对1聊天输入不需要分词。
知识输入如下，需要分词。
```json
{
  "goal": "[1] 问答 ( User 主动 按 『 参考 知识 』   问   『 周迅 』   的 信息 ， Bot 回答 ， User 满意 并 好评 )--> ...... --> [3] 电影 推荐 ( Bot 主动 ， Bot 使用   『 李米的猜想 』   的 某个 评论 当做 推荐 理由 来 推荐   『 李米的猜想 』 ， User 先问 电影 『 国家 地区 、 导演 、 类型 、 主演 、 口碑 、 评分 』 中 的 一个 或 多个 ， Bot 回答 ， 最终 User 接受 ) --> [4] 再见"
  "situation": "聊天 时间 : 中午 12 : 00 ， 在 学校     聊天 主题 : 考试 不好", 
  "user_profile": 
  {
    "姓名": "柳鑫彬", 
    "性别": "男", 
    "居住地": "衡水", 
    "年龄区间": "18-25", 
    "职业状态": "学生", 
    "喜欢 的 明星": ["黄渤"], 
    "喜欢 的 电影": ["天才眼镜狗"], 
    "喜欢 的 兴趣点": ["大肚雪花饺子"], 
    "同意 的 美食": " 烤鱼", "同意 的 新闻": " 黄渤 的新闻", 
    "拒绝": ["音乐"], 
    "接受 的 电影": [], 
    "没有接受 的 电影": []
  }, 
  "knowledge": 
    [["柳鑫彬", "喜好", "明星"], 
    ["柳鑫彬", "喜好", "电影"], 
    ["柳鑫彬", "喜好", "兴趣点"]], 
  "history": 
    ["[1] 最近 怎么样 呢 ？"]
}
```
# requirement
```
python=3.6.10
torch=1.0.0
transformers=2.5.1
jieba=0.42.1
```
