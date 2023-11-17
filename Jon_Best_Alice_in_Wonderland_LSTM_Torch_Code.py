# -*- coding: utf-8 -*-
"""
Created on Sun Jul 9 08:41:52 2023

@author: JonBest
"""

# Jon Best
# Advanced Machine Learning Examples  
# The purpose of this Python code is to implement a text-generated long short-term memory algorithm using
# the PyTorch neural network library and the model to generate new text based on Alice in Wonderland.
 
#***************************************************************************************
# Title: Create your first Text Generator with LSTM in few minutes
# Author: Editorial Team
# Date: 2020
# Availability: https://towardsai.net/p/deep-learning/create-your-first-text-generator-with-lstm-in-few-minutes#83cd
#
# Title: NLP with PyTorch: A Comprehensive Guide
# Author: Moez Ali
# Date: 2023
# Availability: https://www.datacamp.com/tutorial/nlp-with-pytorch-a-comprehensive-guide
#
# Title: Generating WordClouds in Python Tutorial 
# Author: Duong Vu
# Date: 2023
# Availability: https://www.datacamp.com/tutorial/wordcloud-python
#
# Title: POS Tagging with NLTK and Chunking in NLP
# Author: Daniel Johnson
# Date: 2023
# Availability: https://www.guru99.com/pos-tagging-chunking-nltk.html
#
# Title: Text Generation with LSTM in PyTorch
# Author: Adrian Tam
# Date: 2023
# Availability: https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/
#
# Title: Text generation with LSTM in Pytorch
# Author: N. Arvand
# Date: 2023
# Availability: https://www.kaggle.com/code/arviinndn/text-generation-with-lstm-in-pytorch
#
# Title: Using LSTM in PyTorch: A Tutorial With Examples
# Author: Saurav Maheshkar
# Date: 2023
# Availability: https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5
#
# Title: Working with Networkx
# Author: Aditya Singh
# Date: 2022
# Availability: https://medium.com/@aditya26sg/working-with-networkx-382af2c3ec1
#
#***************************************************************************************

# Imported libraries include: torch for machine learning framework, numpy for advanced mathematical functions, 
# nltk for natural language toolkit support, re for regular expression support, wordcloud to create a wordcloud graphic,
# and matplotlib for graphic representation of data.
import torch
import torch.nn as nn
import numpy as np
import nltk
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the Alice in Wonderland text dataset
with open("alice_in_wonderland.txt", "r") as file:
    text = file.read()

# Set the size of the subset to the first 100,000 characters.
subset_size = 100000
text_subset = text[:subset_size]

# Create a character-level vocabulary of unique characters and the corresponding indices.
chars = sorted(list(set(text_subset)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
vocab_size = len(chars)

# Convert the text to numerical input to be used for representations during training and text generation tasks.
input_seq = [char_to_idx[char] for char in text_subset]

# Prepare the input and target data sequences of fixed length.
seq_length = 100
input_data = []
target_data = []
for i in range(len(input_seq) - seq_length):
    input_data.append(input_seq[i:i+seq_length])
    target_data.append(input_seq[i+1:i+seq_length+1])

# Convert the input and target data to numpy arrays.
input_data = np.array(input_data)
target_data = np.array(target_data)

# Convert numpy arrays to PyTorch tensors.
input_tensor = torch.from_numpy(input_data).long()
target_tensor = torch.from_numpy(target_data).long()

# Define the LSTM text-based generation model.
class LSTMTextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMTextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)
        output = self.fc(output)
        return output

# Hyperparameters used to influence the behavior and performance of the LSTM text generation model during training.
hidden_size = 128
num_epochs = 25
batch_size = 64
learning_rate = 0.001

# Create the data loader to handle large datasets and load training data in batchs.
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the LSTM model.
model = LSTMTextGenerator(vocab_size, hidden_size, vocab_size)

# Loss and optimizer - criterion to used for multiclass problems and optimizer used adaptive learning rate optimization algorithm.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop to train model over multiple epochs.
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Forward pass sends inputs to the model.
        outputs = model(inputs)

        # Compute loss using scaler value between predicted outputs and the ground truth targets.
        loss = criterion(outputs.transpose(1, 2), targets)

        # Backward and optimize - performs backpropagation and updates model parameters. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Display the current epoch number and the corresponding loss value.
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}") 
    
# Line graph used to track loss values during training.
loss_values = []

# Training loop.
for epoch in range(num_epochs):
    
    # Compute loss and perform optimization.
    loss_values.append(loss.item())

# Plot the loss curve.
plt.figure(figsize=(12, 8))
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
plt.close()

# Display word frequencies visual bar graph.
# Generate word frequencies.
word_frequencies = nltk.FreqDist(text.split())

# Get the most common words and their frequencies.
most_common_words = word_frequencies.most_common(20)
words, frequencies = zip(*most_common_words)

# Create a bar chart.
plt.figure(figsize=(12, 8))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Most Common Words in Alice in Wonderland')
plt.xticks(rotation='vertical')
plt.show()
plt.close()

# Generate word cloud.
wordcloud = WordCloud().generate(text)

# Display the word cloud.
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
plt.close()

# Split the text into paragraphs
paragraphs = re.split('\n{2,}', text)

# Calculate word count for each paragraph.
word_counts = [len(nltk.word_tokenize(paragraph)) for paragraph in paragraphs]

# Plot the word count line graph
plt.figure(figsize=(12, 8))
plt.plot(word_counts)
plt.xlabel('Paragraph')
plt.ylabel('Word Count')
plt.title('Word Count by Paragraph')
plt.show()

#Display word length distribution graph.
word_lengths = [len(word) for word in text.split()]
plt.figure(figsize=(12, 8))
plt.hist(word_lengths, bins=20)
plt.xlabel('Word Length')
plt.ylabel('Count')
plt.title('Word Length Distribution')
plt.show()
plt.close()

# Display character frequency visual graph.
char_frequencies = nltk.FreqDist(text)
char_counts = [char_frequencies[char] for char in chars]
plt.figure(figsize=(12, 8))
plt.bar(chars, char_counts)
plt.xlabel('Character')
plt.ylabel('Frequency')
plt.title('Character Frequency')
plt.xticks(rotation='vertical')
plt.show()
plt.close()

# Display sentence distribution visual graph.
sentences = nltk.sent_tokenize(text)
sentence_lengths = [len(sentence.split()) for sentence in sentences]
plt.figure(figsize=(12, 8))
plt.hist(sentence_lengths, bins=20)
plt.xlabel('Sentence Length')
plt.ylabel('Count')
plt.title('Sentence Length Distribution')
plt.show()
plt.close()

# Bigram Frequency Network visual graph.
# Compute bigram frequencies.
bigrams = list(nltk.bigrams(text.split()))
bigram_frequencies = nltk.FreqDist(bigrams)

# Specify the number of most frequent bigrams to display.
num_top_bigrams = 20

# Get the top N most frequent bigrams and their frequencies.
top_bigrams = bigram_frequencies.most_common(num_top_bigrams)
bigram_labels = [' '.join(bigram) for bigram, _ in top_bigrams]
bigram_counts = [count for _, count in top_bigrams]

# Plot the most frequent bigrams.
plt.figure(figsize=(10, 6))
plt.bar(bigram_labels, bigram_counts)
plt.xlabel('Bigram')
plt.ylabel('Frequency')
plt.title(f'Top {num_top_bigrams} Most Frequent Bigrams')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()
plt.close()

# Display Part-of-Speech Distribution visual graph.
pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
pos_counts = nltk.FreqDist(tag for _, tag in pos_tags)

plt.figure(figsize=(12, 8))
plt.bar(pos_counts.keys(), pos_counts.values())
plt.xlabel('Part of Speech')
plt.ylabel('Count')
plt.title('Part-of-Speech Distribution')
plt.xticks(rotation='vertical')
plt.show()
plt.close()

# Generate text using the trained LSTM text generation model.
with torch.no_grad():
    seed_text = "Alice"
    input_eval = torch.LongTensor([[char_to_idx[char] for char in seed_text]])
    generated_text = seed_text
    
    # Specify the maximum length of the generated text.
    max_length = 500  

    for _ in range(max_length):
        output = model(input_eval)
        probabilities = nn.functional.softmax(output, dim=2)
        predicted_idx = torch.multinomial(probabilities[:, -1, :], num_samples=1)
        predicted_char = chars[predicted_idx.item()]
        generated_text += predicted_char
        input_eval = torch.LongTensor([[char_to_idx[predicted_char]]])

print(generated_text)
