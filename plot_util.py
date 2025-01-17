import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = []
    val_acc = []
    if 'acc' in history.history:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
    elif 'accuracy' in history.history:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    elif 'categorical_accuracy' in history.history:
        acc = history.history['categorical_accuracy']
        val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('results.png')
    print('Saved results plot to disk')