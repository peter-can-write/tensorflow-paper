with tf.name_scope('Variables'):
    x = tf.constant(1.0)
    y = tf.constant(2.0)
    tf.scalar_summary('z', x + y)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('/tmp/log', graph)

with tf.Session(graph=graph):
    for step in range(1000):
        writer.add_summary(
            merged.eval(), global_step=step)
