# Kaggle_San_Francisco_Crime_Classification
SVC处理多维度数据太慢，据说是二次规划问题？总之是很废

RF训练得分2.6，计算相对快速 （bagging方法，实际效果需等提交）

- to do：
1. xgboost/lightgbm待测 (boosting方法，讲道理应该比bagging牛逼一点，但是涉及梯度下降，训练会慢一些)
2. stacking: 或可针对RF进行优化，对每棵树的输出做训练，改变投票权重。
3. 手撕代码！就是刚！

