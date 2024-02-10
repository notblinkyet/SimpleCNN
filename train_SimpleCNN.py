import SimpeCNN from SimpleCNN
import train from train

DEVICE = torch.device("cuda")

model = SimpleCNN()

history3 = train(train_dataset, val_dataset, model=model.to(DEVICE), epoches=2, batch_size=64)

loss, acc, val_loss, val_acc = zip(*history3)

plt.figure(figsize=(15, 9))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
