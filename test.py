acc = 0
for input_data, output_data in zip(xtest, ytest):
  output = obj.forwardPass(input_data)
  if output[output_data] >= 0.9:
    acc+=1
print(acc/len(ytest))
