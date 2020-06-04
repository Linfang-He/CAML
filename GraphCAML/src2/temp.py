def test_evaluate(self):
    print("Evaluting test dataset......")
    self.mdl.saver.restore(self.sess, '../output/model_best_' + str(DATACNT) + '.ckpt')

    all_preds = []
    losses = []
    actual_labels = [float(line[2]) for line in self.test_rating_set]  # 实际评分

    start = datetime.datetime.now()
    # batch_size设置为1，方便计算每一个gen_reivew和source_review的bleu和rouge

    self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false))  # 不训练模型
    for i in tqdm(range(self.test_len)):
        batch = batchify(self.test_rating_set, self.test_reviews, i, 1, max_sample=self.test_len)

        batch = self._prepare_set(batch)
        if (len(batch) == 0):
            continue

        feed_dict = self.mdl.get_feed_dict(batch, mode='testing')

        preds = self.sess.run([self.mdl.output_pos], feed_dict)

        all_preds += [x[0] for x in preds]  # preds 预测评分
        # losses.append(loss)

    print("\n预测分数为小数时：")
    # all_preds 是0~1之间的数
    all_preds = [rescale(x) for x in all_preds]
    mse = mean_squared_error(actual_labels, all_preds)
    mae = mean_absolute_error(actual_labels, all_preds)
    print("MAE: ", mae)
    print("RMSE: ", mse ** 0.5)  # 开根号

    print("\n预测分数四舍五入转化为整数时：")
    all_preds = [int(x + 0.5) for x in all_preds]
    actual_labels = [int(x) for x in actual_labels]
    mse_int = mean_squared_error(actual_labels, all_preds)
    mae_int = mean_absolute_error(actual_labels, all_preds)
    print("MAE_int: ", mae_int)
    print("RMSE_int: ", mse_int ** 0.5)