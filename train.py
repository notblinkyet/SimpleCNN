from tqdm import tqdm, tqdm_notebook

device = torch.device("cuda")
def epoch_fit(model, train_loader, criterion, optimizer):
  running_loss = 0.0
  running_correct = 0
  processed_data = 0

  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    preds = torch.argmax(outputs, 1)
    running_loss += loss.item() * inputs.size(0)
    running_correct += torch.sum(preds == labels.data)
    processed_data += inputs.size(0)

  train_loss = running_loss / processed_data
  train_acc = running_correct.cpu().numpy() / processed_data
  return train_loss, train_acc

def eval_fit(model, val_loader, criterion):
  model.eval()
  running_loss = 0.0
  running_correct = 0
  processed_size = 0

  for inputs, labels in val_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      preds = torch.argmax(outputs, 1)

    running_loss += loss.item() * outputs.size(0)
    running_correct += torch.sum(preds == labels)
    processed_size += inputs.size(0)
  val_loss = running_loss / processed_size
  val_acc = running_correct.double() / processed_size
  return val_loss, val_acc

def train(train_dataset, val_dataset, model, epoches, batch_size):
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  history = []

  with tqdm(desc="epoch", total=epoches) as pbar_outer:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoches):
      train_loss, train_acc = epoch_fit(model,train_loader, criterion, opt)
      print("loss", train_loss)

      val_loss, val_acc = eval_fit(model, val_loader, criterion)
      history.append((train_loss, train_acc, val_loss, val_acc))

      pbar_outer.update(1)
      tqdm.write(f"\nEpoch {epoch+1} train_loss: {train_loss} \
    val_loss {val_loss} train_acc {train_acc} val_acc {val_acc}")
  return history
