import tensorflow as tf
import os
import numpy as np

from model import Transformer
from preprocess import preprocessdata
import tqdm


def genpaddingmask(data):
    return tf.cast(data == 0, tf.float32)[:, tf.newaxis, tf.newaxis, :]


def genlookmask(size):
    mat = (1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0))[
        tf.newaxis, tf.newaxis, :, :
    ]

    # tf.print(mat[0, 0, :5, :5])

    return mat


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class HyperParameters:
    def __init__(self, data):
        self.num_layers = 4
        self.d_model = 128
        self.dff = 512
        self.num_heads = 8
        self.dropout_rate = 0.1

        self.cn_vocab_size = data.cntok.vocab_size + 2
        self.en_vocab_size = data.entok.vocab_size + 2


class CatetoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name):
        super().__init__(name=name, dtype=tf.float32)
        self._total = self.add_weight("total", initializer=tf.zeros_initializer)
        self._count = self.add_weight("count", initializer=tf.zeros_initializer)

    def reset_states(self):
        self._total.assign(0)
        self._count.assign(0)

    def result(self):
        return self._count / self._total

    def update_state(self, real, pred):
        pred = tf.math.argmax(pred, axis=-1)

        nonpad = real != 0
        eq = real == pred
        count = tf.math.reduce_sum(tf.cast(nonpad, tf.float32))
        countcorr = tf.math.reduce_sum(
            tf.cast(tf.math.logical_and(nonpad, eq), tf.float32)
        )

        self._total.assign_add(count)
        self._count.assign_add(countcorr)


def main():
    data = preprocessdata()
    params = HyperParameters(data)
    learning_rate = CustomSchedule(params.d_model)

    model = Transformer(
        params.num_layers,
        params.d_model,
        params.num_heads,
        params.dff,
        params.cn_vocab_size,
        params.en_vocab_size,
        params.cn_vocab_size,
        params.en_vocab_size,
        params.dropout_rate,
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = CatetoricalAccuracy(name="accuracy")

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, "train/model", max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    @tf.function
    def train_step(en, cn):
        en_input = en[:, :-1]
        en_output = en[:, 1:]

        input_mask = genpaddingmask(cn)
        decode_mask = tf.maximum(
            genpaddingmask(en_input), genlookmask(tf.shape(en_input)[1])
        )

        with tf.GradientTape() as tape:
            pred, _ = model(cn, en_input, True, input_mask, decode_mask, input_mask)
            loss = loss_object(en_output, pred)
            loss = loss * tf.cast(en_output != 0, tf.float32)
            loss = tf.reduce_mean(loss)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        train_accuracy(en_output, pred)

    @tf.function
    def valid_step(en, cn):
        en_input = en[:, :-1]
        en_output = en[:, 1:]
        input_mask = genpaddingmask(cn)
        decode_mask = tf.maximum(
            genpaddingmask(en_input), genlookmask(tf.shape(en_input)[1])
        )
        pred, _ = model(cn, en_input, False, input_mask, decode_mask, input_mask)
        train_accuracy(en_output, pred)

    maxtrainacc = 0

    for epoch in range(1000):
        train_loss.reset_states()
        train_accuracy.reset_states()
        t = tqdm.tqdm(enumerate(data.data_train))
        for batch, (en, cn) in t:
            train_step(en, cn)
            t.desc = f"{epoch:03d}: {train_loss.result():.4f}, {train_accuracy.result():.4f}"

        t = tqdm.tqdm(enumerate(data.data_validation))
        for batch, (en, cn) in t:
            valid_step(en, cn)
            t.desc = f"{epoch:03d}: {train_accuracy.result():.4f}"
        print(f'Epoch {epoch:03d}: {train_accuracy.result():.4f}')

        ckpt_save_path = ckpt_manager.save()

        curtrainacc = float(train_accuracy.result())
        if curtrainacc > maxtrainacc:
            os.system("rm -rf train-best")
            os.system("cp -r train train-best")
            maxtrainacc = curtrainacc
            print("Copy Best")

        

        # print(
        #     "Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
        #         epoch + 1, train_loss.result(), train_accuracy.result()
        #     )
        # )


def evaluation():
    data = preprocessdata()
    params = HyperParameters(data)

    model = Transformer(
        params.num_layers,
        params.d_model,
        params.num_heads,
        params.dff,
        params.cn_vocab_size,
        params.en_vocab_size,
        params.cn_vocab_size,
        params.en_vocab_size,
        params.dropout_rate,
    )

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, "train-best/model", max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    NUM_BEAM = 4

    @tf.function
    def eval_step(en, cn, idx):
        en_input = en
        input_mask = genpaddingmask(cn)
        decode_mask = tf.maximum(
            genpaddingmask(en_input), genlookmask(tf.shape(en_input)[1])
        )

        pred, _ = model(cn, en_input, False, input_mask, decode_mask, input_mask)
        pred = tf.nn.softmax(pred, -1)
        values, indices = tf.math.top_k(pred, NUM_BEAM)
        return values, indices

    SEQLEN = 20

    total = 0
    count = 0

    for batch, (en, cn) in enumerate(data.data_test):
        en_pred = np.zeros(shape=(int(tf.shape(cn)[0]), NUM_BEAM, SEQLEN), dtype=np.int)
        en_pred[:, :, 0] = data.entok_begin
        for outlab in range(SEQLEN - 1):
            values, indices = eval_step(tf.convert_to_tensor(en_pred[:, :-1]), cn, outlab)

        for i, x in enumerate(en_pred):
            for j in range(SEQLEN - 1):
                if x[j] == data.entok_end:
                    x[j + 1 :] = 0
                    break

        en_ground = en[:, 1:]
        en_output = en_pred[:, 1:]

        nonpad = np.logical_or(en_ground != 0, en_output != 0)
        eq = np.logical_and(nonpad, en_ground == en_output)

        total += np.sum(nonpad)
        count += np.sum(eq)


        for idx in range(10):
            x = en_ground[idx].numpy()
            y = en_output[idx]
            l = cn[idx, 1:].numpy()
            for j in range(SEQLEN-1):
                if l[j] == data.cntok_end:
                    l[j:] = 0
            for j in range(SEQLEN-1):
                if x[j] == data.entok_end:
                    x[j:] = 0
            for j in range(SEQLEN-1):
                if y[j] == data.entok_end:
                    y[j:] = 0
            print(data.cntok.decode(l))
            print(data.entok.decode(x))
            print(data.entok.decode(y))
            print()
        break

    print(count / total)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # main()
    evaluation()
