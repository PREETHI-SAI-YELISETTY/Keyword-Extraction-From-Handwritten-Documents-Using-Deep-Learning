# Keyword-Extraction-From-Handwritten-Documents-Using-Deep-Learning

# MOTIVATION
Extensive research has gone into the problem of handwritten text recognition and classification. Starting with
OCR softwares to Deep learning. DL techniques have greatly improved the rate of Handwriting recognition.
Neural Networks are used to recognise the text. However there are still some limitations that lead to inaccurate
transcriptions of text. Specifically when it comes to Offline Handwriting Recognition that involves historical
documents, archives, hand-filled forms etc. For such situations we propose Keyword Extraction of the
recognised text, recognised using a combination of CNN, RNN and CTC layers, as it can be a viable solution to
reduce the inaccuracies while also providing a low-cost way to index and search the documents. We propose
Bi-LSTM model for keyword extraction.
# KEYWORDS: 
Offline Handwriting recognition, Historical documents, Deep Learning, Keyword Extraction, CNN, RNN, CTC, Bi-LSTM
# INTRODUCTION:
Handwriting recognition, a subfield of OCR technology, is the process of converting handwritten, analog
documents into digital text. It is defined as the ability of a computer to receive and interpret intelligible
handwritten input from sources such as paper documents, photographs, touch-screens, and other devices.A
handwriting recognition system handles formatting, performs correct segmentation into characters, and finds the
most plausible words.
This involves mainly two steps:
(i) Character Extraction:
This involves extracting the individual characters from an image of the handwritten document. We must
consider the case of connected characters and perform proper sentence segmentation to accurately extract the
characters.
(ii) Feature Extraction:
Individual properties of the different characters are hard encoded into the system and the input symbols,
extracted from images, are matched.Properties include aspect ratio, pixel distribution, number of strokes,
distance from the image centre, and reflection. This is computationally intensive and prone to inaccuracies.
Modern techniques concentrate on recognising all the characters in a segmented line of text, as opposed to
previous strategies that concentrate on segmenting individual characters for recognition. The current focus is on
machine learning and deep learning techniques that help overcome time and computation restrictions. A text
analysis approach called keyword extraction mechanically extracts the most frequently used and significant
words and expressions from a text. It helps summarise the content of texts and recognize the main topics
discussed. Keyword extraction methods include TF-IDF, RAKE, and ML approaches such as SVMs and LSTM
models.

# PROPOSED METHOD:
For the HWR we propose a Neural Network model that consists of a CNN, RNN and CTC layer.
CNN: The layers of CNN are fed the input image. These layers have been taught to extract from the image the
necessary features. Every layer has three operations. First, there is the convolution process, which applies a filter
kernel to the input that is 55 in size for the first two layers and 33 for the final three layers. The non-linear
Rectified Linear Unit (RELU) function is then utilized. A pooling layer then outputs a condensed version of the
input after summarizing image regions.
RNN: The RNN propagates pertinent information through the feature sequence, which has 256 features per
time-step. The well-known Long Short-Term Memory (LSTM) implementation of RNNs is chosen since it has
more robust training features than a vanilla RNN and can transmit information over greater distances. A matrix
with a dimension of 3280 is used to map the RNN output sequence.
CTC: While the NN is being trained, the CTC calculates the loss value using the ground truth text and the RNN
output matrix. The CTC only receives the matrix during inference, which it uses to decode into the final text.
The maximum character length for the recognised text and the ground truth text is 32.The output of the HWR
model is then fed to a Keyword extraction model as input. We use Bi-LSTM for Keyword Extraction.
Bi-LSTM: Similar to conventional network forward and backward passes, the forward and backward passes
over the unfolded network over time are performed, with the exception that we must unfold the concealed states
at each time step. Additionally, the start and end of the data points require careful attention. We just need to reset
the hidden states to 0 at the beginning of each sentence in our implementation because we move forward and
backward for entire sentences. We can process numerous sentences at once thanks to our batch implementation.
