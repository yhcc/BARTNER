from fastNLP.core.callback import Callback
from fastNLP import DataSet, Tester
import fitlog
from copy import deepcopy


class FitlogCallback(Callback):
    r"""
    该callback可将loss和progress写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
    一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
    并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
    fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=True,
                 raise_threshold=0, better_dev_eval=True, eval_begin_epoch=-1):
        r"""

        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要
            传入多个DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。data的结果的名称以'data'开头。
        :param ~fastNLP.Tester,Dict[~fastNLP.Tester] tester: Tester对象，将在on_valid_end时调用。tester的结果的名称以'tester'开头
        :param int log_loss_every: 多少个step记录一次loss(记录的是这几个batch的loss平均值)，如果数据集较大建议将该值设置得
            大一些，不然会导致log文件巨大。默认为0, 即不要记录loss。
        :param int verbose: 是否在终端打印evaluation的结果，0不打印。
        :param bool log_exception: fitlog是否记录发生的exception信息
        :param float raise_threshold: 如果metric值低于这个就会raise exception
        :param bool better_dev_eval: 仅当dev取得更好的结果才做evaluate
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        self.raise_threshold = raise_threshold
        self.eval_begin_epoch = eval_begin_epoch

        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.better_dev_eval = better_dev_eval

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.trainer.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.kwargs.get('test_use_tqdm', self.trainer.use_tqdm),
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

        if self.trainer.save_path is not None:
            model_name = "best_" + "_".join([self.model.__class__.__name__, self.trainer.metric_key, self.trainer.start_time])
            fitlog.add_other(name='model_name', value=model_name)

    def on_epoch_begin(self):
        if self.eval_begin_epoch>0 and self.epoch>self.eval_begin_epoch:
            self.trainer.validate_every = -1

    def on_backward_begin(self, loss):
        if self._log_loss_every >0:
            self._avg_loss += loss.item()
            if self.step %self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss /self._log_loss_every *self.update_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        indicator, indicator_val = _check_eval_results(eval_result, metric_key=metric_key)
        if indicator_val < self.raise_threshold:
            raise RuntimeError("The program has been running off.")

        if len(self.testers) > 0:
            do_eval = True
            if self.better_dev_eval:
                if not better_result:
                    do_eval = False
            if do_eval:
                for idx, (key, tester) in enumerate(self.testers.items()):
                    try:
                        eval_result = tester.test()
                        if self.verbose != 0:
                            self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                            self.pbar.write(tester._format_eval_results(eval_result))
                        fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                        if idx == 0:
                            indicator, indicator_val = _check_eval_results(eval_result, metric_key=self.trainer.metric_key)
                            if indicator_val>self.best_test_metric_sofar:
                                self.best_test_metric_sofar = indicator_val
                                self.best_test_epoch = self.epoch
                                self.best_test_sofar = eval_result

                        if better_result:
                            self.best_dev_test = eval_result
                            self.best_dev_epoch = self.epoch
                            fitlog.add_best_metric(eval_result, name=key)
                    except Exception as e:
                        self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                        raise e

    def on_train_end(self):
        if self.best_test_sofar:
            line1 = "Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_test_sofar, self.best_test_epoch)
            self.logger.info(line1)
            fitlog.add_to_line(line1)
        if self.best_dev_test:
            line2 = "Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_dev_test, self.best_dev_epoch)
            self.logger.info(line2)
            fitlog.add_to_line(line2)
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


def _check_eval_results(metrics, metric_key=None):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val


from fastNLP import WarmupCallback as FWarmupCallback
import math
class WarmupCallback(FWarmupCallback):

    def __init__(self, warmup=0.1, schedule='constant'):
        """

        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_inverse_square_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)


class OutputIndiceCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_batch_begin(self, batch_x, batch_y, indices):
        self.indices = indices

    def on_exception(self, exception):
        print(self.indices)