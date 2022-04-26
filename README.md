# sohu-2022-competition
 2022搜狐校园 情感分析×推荐排序 算法

### **推荐**

##### 数据预处理

- 通过`pvId`随机划分 train & valid，比例19:1

##### 特征构造
1. `suv`
2. `itemId`
3. `city`
4. `hour`: categorical_column_with_identity, 24
5. `min`: categorical_column_with_identity, 1440
6. `logTs`: categorical_column_with_hash_bucket, 1000
7. `osType`: categorical_column_with_hash_bucket, 11
8. `deviceType`: categorical_column_with_hash_bucket, 4
8. `province`: categorical_column_with_hash_bucket, 30
8. `seqLen`&`sum`: categorical_column_with_hash_bucket, 300,2000


##### 结果

直接切分train & valid

| 使用特征 | 测试成绩 | 线上成绩 |
| :------- | :------- | :------- |
| 1-3      | 61.555   | 53.000   |
| 1-4      | 60.167   | 53.495   |

按`pvId`随机划分train & valid

| 使用特征          | 测试成绩 | 线上成绩 |
| :---------------- | :------- | :------- |
| 1-3               | 54.484   | 53.000   |
| +hour             | 54.689   | None     |
| +min              | 54.286   | None     |
| +logTs            | 54.251   | None     |
| +osType           | 54.299   | None     |
| +deviceType       | 54.792   | None     |
| +hour +deviceType | 56.166   | 54.331   |
| +seqLen&sum       | 57.340   | None     |
| +province         | 54.847   | None     |
| 以上所有有效特征  | 59.407   | 57.372   |
