f = 1000;
A = 10;
t1 = 0:0.001:0.05;
t2 = 0:0.0005:0.05;
t3 = 0:0.0001:0.05;
L = 4;

%Sampling

subplot(3,1,1)
x1 = A*sin(2*pi*f*t1);
plot(t1,x1)
xlabel("Time(s)")
ylabel("Amplitude")

subplot(3,1,2)
x2 = A*sin(2*pi*f*t2);
plot(t2,x2)
xlabel("Time(s)")
ylabel("Amplitude")

subplot(3,1,3)
x3 = A*sin(2*pi*f*t3);
plot(t3,x3)
xlabel("Time(s)")
ylabel("Amplitude")




% uniform quantizer
function [quantized_signal, quantization_error] = uniform_quantizer(x, L)
    % Input: 
    % x - Input signal
    % L - Number of quantization levels

    % Determine input range
    xmax = max(abs(x)); % Maximum amplitude of the signal
    delta = 2 * xmax / L; % Step size

    % Quantization levels
    levels = linspace(-xmax + delta/2, xmax - delta/2, L);

    % Quantize the signal
    quantized_signal = delta * round(x / delta);
    
    % Clip values to within range
    quantized_signal = min(max(quantized_signal, -xmax), xmax);

    % Quantization error
    quantization_error = x - quantized_signal;
end



%Non unifrom quantizer
function [quantized_signal, quantization_error] = mu_law_quantizer(x, L, mu)
    % Input: 
    % x - Input signal
    % L - Number of quantization levels
    % mu - Compression factor (e.g., 255 for audio)

    % Normalize the input signal
    xmax = max(abs(x));
    x_normalized = x / xmax;

    % Apply μ-law compression
    compressed_signal = sign(x_normalized) .* log(1 + mu * abs(x_normalized)) / log(1 + mu);

    % Uniform quantization on the compressed signal
    [quantized_compressed, ~] = uniform_quantizer(compressed_signal, L);

    % Expand the signal back
    quantized_signal = sign(quantized_compressed) .* ((1 + mu) .^ abs(quantized_compressed) - 1) / mu;

    % Denormalize the quantized signal
    quantized_signal = quantized_signal * xmax;

    % Quantization error
    quantization_error = x - quantized_signal;
end



% Number of quantization levels
L_values = [4, 8, 16]; % Multiple quantization levels
mu = 255; % μ-law compression factor

% Create figure for comparison
figure;

for i = 1:length(L_values)
    L = L_values(i);

    % Apply Uniform Quantizer
    [quantized_uniform, quantization_error_uniform] = uniform_quantizer(x1, L);

    % Apply Non-Uniform μ-law Quantizer
    [quantized_mu, quantization_error_mu] = mu_law_quantizer(x1, L, mu);

    % Plot Original vs Quantized (Uniform)
    subplot(length(L_values), 4, 4*(i-1)+1);
    plot(t1, x1, 'b', 'LineWidth', 1); hold on;
    stairs(t1, quantized_uniform, 'r', 'LineWidth', 1);
    grid on;
    title(['Uniform Quantized Signal (L = ' num2str(L) ')']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend('Original Signal', 'Quantized Signal');

    % Plot Quantization Error (Uniform)
    subplot(length(L_values), 4, 4*(i-1)+2);
    plot(t1, quantization_error_uniform, 'k', 'LineWidth', 1);
    grid on;
    title(['Uniform Quantization Error (L = ' num2str(L) ')']);
    xlabel('Time (s)');
    ylabel('Error Amplitude');

    % Plot Original vs Quantized (Non-Uniform)
    subplot(length(L_values), 4, 4*(i-1)+3);
    plot(t1, x1, 'b', 'LineWidth', 1); hold on;
    stairs(t1, quantized_mu, 'r', 'LineWidth', 1);
    grid on;
    title(['Non-Uniform Quantized Signal (L = ' num2str(L) ')']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend('Original Signal', 'Quantized Signal');

    % Plot Quantization Error (Non-Uniform)
    subplot(length(L_values), 4, 4*(i-1)+4);
    plot(t1, quantization_error_mu, 'k', 'LineWidth', 1);
    grid on;
    title(['Non-Uniform Quantization Error (L = ' num2str(L) ')']);
    xlabel('Time (s)');
    ylabel('Error Amplitude');
end




%Quantization Error Analysis 
function [quantized_signal, maqe, variance_error] = uniform_quantizer_metrics(x1, L)
    % Input:
    % x - Input signal (vector)
    % L - Number of quantization levels

    % Ensure the signal is a column vector
    x = x1(:);

    % Maximum amplitude of the signal
    xmax = max(abs(x));

    % Step size (Δ)
    delta = 2 * xmax / L;

    % Quantize the signal
    quantized_signal = delta * round(x / delta);

    % Clip values to ensure they are within range
    quantized_signal = min(max(quantized_signal, -xmax), xmax);

    % Quantization error
    quantization_error = x - quantized_signal;

    % Mean Absolute Quantization Error (MAQE)
    maqe = mean(abs(quantization_error));

    % Variance of Quantization Error (Experimental)
    variance_error = var(quantization_error);
end

% Define parameters
fs = 1000; % Sampling frequency
t = 0:1/fs:1; % Time vector (1 second)
x = 0.8 * sin(2 * pi * 5 * t); % Test signal: 5 Hz sine wave

% Quantization levels
L_values = [4, 8, 16];

% Initialize results
maqe_results = zeros(1, length(L_values)); % Mean Absolute Quantization Error
variance_results = zeros(1, length(L_values)); % Variance of Quantization Error

% Loop through each quantization level
for i = 1:length(L_values)
    L = L_values(i);
    
    % Apply uniform quantization
    [quantized_signal, quantization_error] = uniform_quantizer(x, L);
    
    % Calculate MAQE (Mean Absolute Quantization Error)
    maqe_results(i) = mean(abs(quantization_error));
    
    % Calculate variance of quantization error
    variance_results(i) = var(quantization_error);
    
    % Display results
    fprintf('For L = %d:\n', L);
    fprintf('  Mean Absolute Quantization Error (MAQE): %.4f\n', maqe_results(i));
    fprintf('  Variance of Quantization Error: %.4f\n', variance_results(i));
    fprintf('\n');
end

% Plot MAQE vs L
figure;
plot(L_values, maqe_results, '-o', 'LineWidth', 1.5);
grid on;
title('Mean Absolute Quantization Error (MAQE) vs. L');
xlabel('Quantization Levels (L)');
ylabel('MAQE');
legend('MAQE');

% Plot Variance vs L
figure;
plot(L_values, variance_results, '-o', 'LineWidth', 1.5);
grid on;
title('Variance of Quantization Error vs. L');
xlabel('Quantization Levels (L)');
ylabel('Variance');
legend('Variance');





%SQNR

% Quantization levels
L_values = [4, 8, 16];

% Initialize arrays to store SQNR values
sqnr_experimental = zeros(1, length(L_values));
sqnr_theoretical = zeros(1, length(L_values));

% Compute SQNR for each L
for i = 1:length(L_values)
    L = L_values(i);

    % Apply uniform quantizer
    [quantized_signal, ~, variance_error] = uniform_quantizer_metrics(x1, L);

    % Signal Power
    signal_power = mean(x1 .^ 2);

    % Quantization Noise Power (Experimental)
    noise_power = variance_error;

    % Experimental SQNR
    sqnr_experimental(i) = 10 * log10(signal_power / noise_power);

end

% Display results
disp('SQNR Results:');
disp('L     Experimental SQNR (dB) ');
for i = 1:length(L_values)
    fprintf('%-5d %-25.2f \n', L_values(i), sqnr_experimental(i));
end

% Plot SQNR
figure;
plot(L_values, sqnr_experimental, '-o', 'LineWidth', 1.5, 'DisplayName', 'Experimental');
grid on;
title('SQNR vs. Number of Levels');
xlabel('Number of Levels (L)');
ylabel('SQNR (dB)');
legend('Location', 'best');



%% Huffman Encoding

function [encoded_signal, dict] = huffman_encode(signal)
    symbols = unique(signal);
    probabilities = histcounts(signal, [symbols, max(symbols) + 1], 'Normalization', 'probability');
    dict = huffmandict(symbols, probabilities);
    encoded_signal = huffmanenco(signal, dict);
end

%% Huffman Decoding

function decoded_signal = huffman_decode(encoded_signal, dict)
    decoded_signal = huffmandeco(encoded_signal, dict);
end
%%


% Apply uniform quantizer
[quantized_uniform, ~] = uniform_quantizer(x1, L);


% Huffman Encoding and Noiseless Channel Simulation
fprintf('Huffman Encoding Results:\n');

% Process Uniform Quantized Signal
[encoded_signal, huffman_dict] = huffman_encode(quantized_uniform);

%Decoding
decoded_signal = huffman_decode(encoded_signal, huffman_dict);


%signal comparison 
figure;
subplot(2, 1, 1);
plot(x1, 'b', 'LineWidth', 1); hold on;
plot(decoded_signal, 'r--', 'LineWidth', 1); % Reconstructed signal
grid on;
title('Input vs Output Signals');
xlabel('Sample Index');
ylabel('Amplitude');
legend('Input Signal', 'Reconstructed Signal');


%cross correlation
[cross_corr, lags] = xcorr(x1, decoded_signal);
subplot(2, 1, 2);
plot(lags, cross_corr, 'k', 'LineWidth', 1);
grid on;
title('Cross-Correlation Between Input and Output Signals');
xlabel('Lag');
ylabel('Cross-Correlation');


%compression rate and efficiency
function compression_rate = calculate_compression_rate(original_bits, compressed_bits)
    compression_rate = (original_bits - compressed_bits) / original_bits;
end

original_bits = length(x1) * ceil(log2(L));
compressed_bits = length(encoded_signal);
compression_rate = calculate_compression_rate(original_bits, compressed_bits);
disp(['Compression Rate: ', num2str(compression_rate * 100), '%']);





%Bonus
% Helper function to simulate the Binary Symmetric Channel (BSC)
function bsc_signal = bsc_channel(encoded_signal, p)
    % Flatten the encoded signal into a bitstream
    bitstream = cell2mat(encoded_signal);
    
    % Flip bits with probability p
    flipped_bits = rand(size(bitstream)) < p;
    bitstream(flipped_bits) = ~bitstream(flipped_bits); % Flip the bits
    
    % Convert back to encoded signal format (same as original)
    bsc_signal = reshape(bitstream, size(encoded_signal));
end

% Helper function to convert encoded signal to bitstream
function bitstream = encode_to_bits(encoded_signal, huff_dict)
    bitstream = [];
    for i = 1:length(encoded_signal)
        % Find Huffman code for each quantized value
        huffman_code = huff_dict{encoded_signal(i)};
        bitstream = [bitstream, huffman_code];
    end
end

% BSC Parameters
p = 0.01; % Probability of bit flip


% Simulate the Binary Symmetric Channel (BSC)
% Introduce bit errors with probability p = 0.01
bsc_signal = bsc_channel(encoded_signal, p);

% Decode the received signal using the Huffman decoder
reconstructed_signal = huffman_decode(bsc_signal, huff_dict, unique_vals);

% Compare the original and reconstructed signals
disp('Original Signal:');
disp(quantized_signal);
disp('Reconstructed Signal after BSC:');
disp(reconstructed_signal);

% Calculate and display the Mean Squared Error (MSE)
MSE = mean((quantized_signal - reconstructed_signal).^2);
disp(['Mean Squared Error (MSE): ', num2str(MSE)]);

% Optionally, calculate the Bit Error Rate (BER)
original_bits = encode_to_bits(encoded_signal, huff_dict); % Convert encoded signal to bits
received_bits = encode_to_bits(bsc_signal, huff_dict); % Convert received (error affected) signal to bits
BER = sum(original_bits ~= received_bits) / length(original_bits);
disp(['Bit Error Rate (BER): ', num2str(BER)]);