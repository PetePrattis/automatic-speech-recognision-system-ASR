# A Python Program / Project

**This is a Python project from my early days as a Computer Science student**

_This program were created for the eighth semester class Voice and Audio Signal Processing
and is final project for the class_

> #### Description of project
>
>>A python script that implements an automatic speech recognision system.
>

> #### Implementation of project
>
> 1. Sampling of the input signal from an initial sampling rate of 22050 to 8000.
> 2. Apply FIR filter to eliminate DC component and 60 Hz buzz. Also, the bandwidth allowed by the passage was limited to the maximum frequency Fn which was defined as half the sampling rate Fs which is in agreement with Nyquist's theorem Fs> = 2 * Fn.
> 3. Definition of 2 important variables L, R, which aim at careful observation of the signal for subsequent extraction of key characteristics of the signal. This is the most basic part that introduces us to Short Time Signal Analysis. In more detail, the variable L is the length of a frame frame in the signal, in order to process it, as previously defined. Within this context, we also define the variable R which is the sliding step within that context. These numbers are not random as, according to the literature, an "effective" window length L is set at 10-40 ms with a slip step R at 10 ms.
> 4. The spectrogram of the signal which we found as follows was useful:
>> - Calculate stft (Short Time Fourier Transform) for each frame window
>> - Converting coefficients to db (decibels) ie stft (db log10Y - amplitude_to_db command)
> 
> 5. Calculate the passage rate from 0 for the signal, taking into account the frame length L with an R slide and then multiply it by 100 to display a percentage. More analytically for the rate of passage of 0, it reflects the change of algebraic sign in successive samples. The rate at which this change occurs is a mere measurement of the frequency content of a signal. 
> 6. Through observations it has become known that in the non-compliant parts of the signal the rate of passage from 0 is quite high compared to the compliant parts.
>> - However, the above measurement is not sufficient to properly distinguish between the singular and non-partial signal, so the short energy time of the signal was calculated and calculated. More specifically, it reflects the variation in width. In a typical speech signal, we can detect significant changes over time. The short-time energy, defined as the sum of the squares of the signals, is calculated, as in the passage rate of 0, taking into account the given frame length L and the slip step R. Indeed, the short-time energy shows high values for consonant parts of the signal, compared to non-consonants that show significantly lower.
>
> 7. The way we solved the problem of discriminating between vowel and vowel was the onset_detect function of the librosa library. More details of the method we implemented in conjunction with the above function are:
>> - Creating an inverted signal
>> - Using the onset_detect function to specify the frequency, frame length, slip step, and activation of the backtracking function for best results, in the original signal and in the inverted
>> - Convert the result from frames to times and samples
>> - In the case of the original signal we find the onset points where the sound begins. But in the case of the inverted file these points are essentially the end of each word. These points, in order to be able to use them to make a definitive assessment of the distinction between the singular and non-symmetric parts of the signal, is sufficient for each of the signals to subtract the onset points, which are in sec form, from the total duration of the signal
>> - Finally, converting the results into samples to create consistent parts of the signal (digits)
>> - View spectroscopy with the resulting Onset points and distinguish between the singular and non-partial signal
>
> 8. After the input signal has been processed, the voice signal recognition process begins. Since our system wants to identify the numbers from 0 to 9, we have secured a database containing 3 speakers who speak from 0-9. This makes our system not so reliable because, according to the literature, if we could have 100 speakers who would record 10 times each digit, then we would have a powerful set of training that provides accurate and reliable word recognition models- digits. At this point it should be mentioned that the whole process was also subject to training
> 9. Then the steps were the following for each digit found:
>> - For subsequent recognition of the speech signal in text, the labels table containing the name (digit) of each speech signal from the training set was created
>> - Finding 13th MFCC's (Mel-Frequency Cepstral Coefficients) for each separate input signal (digit)
>> - Logging of MFCC coefficients (amplitude_to_db - command) for the input signal
>
> 10. Within a repetitive loop that ends after all the speech signals from the training set have been examined:
>> - Find 13 MFCC's for each signal (processed) from the training set
>> - Logging of MFCC coefficients (amplitude_to_db - command)
>> - Implementation of DTW (Dynamic Time Wrapping) algorithm giving MFCC coefficients for the input signal pair and signal from the training set as signal characteristics. We also used Euclidean distance to find the distance between pairs, as well as backtracking.
>> - Finally, in a Dnew table we add the minimum cost we found for each signal pair (input and training), as in the mfccs table the MFCC coefficients set for each signal from the training set. The purpose of the above two tables is to highlight at the end the digit recognized (via Dnew) and the graph showing the similarity line - optimal path (with mfccs contribution)
>
> 11. Then, it is necessary to evaluate the results of the DTW algorithm and extract a recognition percentage. After we output this result, we then display the final recognition of the text input signal.

> #### About this project
>
> - The comments to make the code understandable, are within the .py archive
> - This project was written in IDLE, Python’s Integrated Development and Learning Environment.
> - This program runs for Python version 2.7
> - This repository was created to show the variety of the work I did and experience I gained as a student
>
