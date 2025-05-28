import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hardlim(x):
    return np.heaviside(x, 1)

def Scale_feature(Input, Saturating_threshold, ratio):
    # 计算最小值和最大值
    Min_value = np.min(Input)
    Max_value = np.max(Input)

    # 计算线性缩放参数
    min_value = Saturating_threshold[0] * ratio
    max_value = Saturating_threshold[1] * ratio
    k = (max_value - min_value) / (Max_value - Min_value)
    b = (min_value * Max_value - Min_value * max_value) / (Max_value - Min_value)

    # 进行线性缩放和饱和处理
    Output = Input * k + b
    Output[Output < min_value] = min_value
    Output[Output > max_value] = max_value

    return Output, k, b


def Scale_feature_separately(Input, Saturating_threshold, ratio):
    nNeurons = Input.shape[1]
    k = np.zeros(nNeurons)
    b = np.zeros(nNeurons)
    Output = np.zeros(Input.shape)
    min_value = Saturating_threshold[0] * ratio
    max_value = Saturating_threshold[1] * ratio

    for i in range(nNeurons):
        Min_value = np.min(Input[:, i])
        Max_value = np.max(Input[:, i])
        k[i] = (max_value - min_value) / (Max_value - Min_value)
        b[i] = (min_value * Max_value - Min_value * max_value) / (Max_value - Min_value)
        Output[:, i] = Input[:, i] * k[i] + b[i]
        Output[Output[:, i] < min_value, i] = min_value
        Output[Output[:, i] > max_value, i] = max_value

    return Output, k, b


def radbas(X, beta=1):
    return np.exp(-beta * X**2)

def tribas(X, c=1):
    return np.maximum(0, 1 - np.abs(X) / c)**2

def rvfl_train(trainX, trainY, testX, testY, option):
        # Set default options
        if 'N' not in option or option['N'] is None:
            option['N'] = 100
        if 'bias' not in option or option['bias'] is None:
            option['bias'] = False
        if 'link' not in option or option['link'] is None:
            option['link'] = True
        if 'ActivationFunction' not in option or option['ActivationFunction'] is None:
            option['ActivationFunction'] = 'radbas'
        if 'seed' not in option or option['seed'] is None:
            option['seed'] = 0
        if 'RandomType' not in option or option['RandomType'] is None:
            option['RandomType'] = 'Uniform'
        if 'mode' not in option or option['mode'] is None:
            option['mode'] = 1
        if 'Scale' not in option or option['Scale'] is None:
            option['Scale'] = 1
        if 'Scalemode' not in option or option['Scalemode'] is None:
            option['Scalemode'] = 1

        np.random.seed(option['seed'])
        # 0-1 coding for the target
        U_trainY = np.unique(trainY)
        nclass = len(U_trainY)
        trainY_temp = np.zeros((len(trainY), nclass))

        # 0-1 coding for the target
        for i in range(nclass):
            idx = trainY == U_trainY[i]
            trainY_temp[idx, i] = 1

        # Generate random weights and biases
        Nsample, Nfea = trainX.shape
        N = option['N']
        if option['RandomType'] == 'Uniform':
            if option['Scalemode'] == 3:
                Weight = option['Scale'] * (np.random.rand(Nfea, N) * 2 - 1)
                Bias = option['Scale'] * np.random.rand(1, N)
            else:
                Weight = np.random.rand(Nfea, N) * 2 - 1
                Bias = np.random.rand(1, N)
        elif option['RandomType'] == 'Gaussian':
            Weight = np.random.randn(Nfea, N)
            Bias = np.random.randn(1, N)
        else:
            raise ValueError('Only Gaussian and Uniform distributions are supported')

        # Repeat bias matrix
        Bias_train = np.tile(Bias, (Nsample, 1))

        # Compute hidden layer input
        H = np.dot(trainX, Weight) + Bias_train

        # Apply activation function based on the specified option
        function_name = option.get('ActivationFunction', '').lower()
        if function_name in ['sig', 'sigmoid']:
            if option.get('Scale', False):
                Saturating_threshold = [-4.6, 4.6]
                Saturating_threshold_activate = [0, 1]
                if option.get('Scalemode', 0) == 1:
                    H, k, b = Scale_feature(H, Saturating_threshold, option.get('Scale'))
                elif option.get('Scalemode', 0) == 2:
                    H, k, b = Scale_feature_separately(H, Saturating_threshold, option.get('Scale'))
                H = sigmoid(H)
        elif function_name in ['sin', 'sine']:
            if option.get('Scale', False):
                Saturating_threshold = [-np.pi / 2, np.pi / 2]
                Saturating_threshold_activate = [-1, 1]
                if option.get('Scalemode', 0) == 1:
                    H, k, b = Scale_feature(H, Saturating_threshold, option.get('Scale'))
                elif option.get('Scalemode', 0) == 2:
                    H, k, b = Scale_feature_separately(H, Saturating_threshold, option.get('Scale'))
                H = np.sin(H)
        elif function_name == 'hardlim':
            H = np.double(hardlim(H))
        elif function_name == 'tribas':
            if option.get('Scale', False):
                Saturating_threshold = [-1, 1]
                Saturating_threshold_activate = [0, 1]
                if option.get('Scalemode', 0) == 1:
                    H, k, b = Scale_feature(H, Saturating_threshold, option.get('Scale'))
                elif option.get('Scalemode', 0) == 2:
                    H, k, b = Scale_feature_separately(H, Saturating_threshold, option.get('Scale'))
                H = tribas(H)
        elif function_name == 'radbas':
            if option.get('Scale', False):
                Saturating_threshold = [-2.1, 2.1]
                Saturating_threshold_activate = [0, 1]
                if option.get('Scalemode', 0) == 1:
                    H, k, b = Scale_feature(H, Saturating_threshold, option.get('Scale'))
                elif option.get('Scalemode', 0) == 2:
                    H, k, b = Scale_feature_separately(H, Saturating_threshold, option.get('Scale'))
                H = radbas(H)
        elif function_name == 'sign':
            H = np.sign(H)


        if option.get('bias', False):
            H = np.hstack((H, np.ones((H.shape[0], 1))))

        if option['link']:
            if option['Scalemode'] == 1:
                trainX_temp = trainX * k + b
                H = np.concatenate((H, trainX_temp), axis=1)
            elif option['Scalemode'] == 2:
                trainX_temp, ktr, btr = Scale_feature_separately(trainX, Saturating_threshold_activate, option['Scale'])
                H = np.concatenate((H, trainX_temp), axis=1)
            else:
                H = np.concatenate((H, trainX), axis=1)

        H[np.isnan(H)] = 0


        # Calculate beta based on option.mode
        if option['mode'] == 2:
            beta = np.linalg.pinv(H) @ trainY_temp
        elif option['mode'] == 1:
            if not hasattr(option, 'C') or option['C'] is None:
                option['C'] = 0.1
            C = option['C']
            if N < Nsample:
                beta = np.linalg.inv(np.eye(H.shape[1]) / C + H.T @ H) @ H.T @ trainY_temp
            else:
                beta = H.T @ np.linalg.inv(np.eye(H.shape[0]) / C + H @ H.T) @ trainY_temp
        else:
            raise ValueError(
                'Unsupport mode, only Regularized least square and Moore-Penrose pseudoinverse are allowed.')

        trainY_temp = np.dot(H, beta)  # H和beta的矩阵乘积
        Y_temp = np.zeros(Nsample)
        # 解码目标
        for i in range(Nsample):
            maxvalue = np.max(trainY_temp[i, :])
            idx = np.argmax(trainY_temp[i, :])
            Y_temp[i] = U_trainY[idx]

        Bias_test = np.tile(Bias, (testY.size, 1))
        H_test = np.dot(testX, Weight) + Bias_test

        if option['ActivationFunction'].lower() in ['sig', 'sigmoid']:
            # Sigmoid
            if option['Scale']:
                if option['Scalemode'] == 1:
                    H_test = H_test * k + b
                elif option['Scalemode'] == 2:
                    kt = np.tile(k, (H_test.shape[0], 1))
                    bt = np.tile(b, (H_test.shape[0], 1))
                    H_test = H_test * kt + bt
            H_test = 1 / (1 + np.exp(-H_test))
        elif option['ActivationFunction'].lower() in ['sin', 'sine']:
            # Sine
            if option['Scale']:
                if option['Scalemode'] == 1:
                    H_test = H_test * k + b
                elif option['Scalemode'] == 2:
                    kt = np.tile(k, (H_test.shape[0], 1))
                    bt = np.tile(b, (H_test.shape[0], 1))
                    H_test = H_test * kt + bt
            H_test = np.sin(H_test)
        elif option['ActivationFunction'].lower() == 'hardlim':
            # Hard limit
            H_test = np.double(H_test > 0)
        elif option['ActivationFunction'].lower() == 'tribas':
            # Triangular basis function
            if option['Scale']:
                if option['Scalemode'] == 1:
                    H_test = H_test * k + b
                elif option['Scalemode'] == 2:
                    kt = np.tile(k, (H_test.shape[0], 1))
                    bt = np.tile(b, (H_test.shape[0], 1))
                    H_test = H_test * kt + bt
            H_test = tribas(H_test)
        elif option['ActivationFunction'].lower() == 'radbas':
            # Radial basis function
            if option['Scale']:
                if option['Scalemode'] == 1:
                    H_test = H_test * k + b
                elif option['Scalemode'] == 2:
                    kt = np.tile(k, (H_test.shape[0], 1))
                    bt = np.tile(b, (H_test.shape[0], 1))
                    H_test = H_test * kt + bt
            H_test = radbas(H_test)
        elif option['ActivationFunction'].lower() == 'sign':
            # Sign
            H_test = np.sign(H_test)


        # H_test: 测试数据矩阵
        # testX: 测试数据特征
        # testY: 测试数据标签
        # U_trainY: 训练数据标签的正交基
        # beta: 线性回归模型的参数向量
        # option: 参数选项

        # 判断是否需要添加偏置项
        if option['bias']:
            H_test = np.concatenate((H_test, np.ones((testY.size, 1))), axis=1)

        # 判断是否需要进行特征缩放操作
        if option['link']:
            if option['Scalemode'] == 1:
                testX_temp = testX * k + b
                H_test = np.concatenate((H_test, testX_temp), axis=1)
            elif option['Scalemode'] == 2:
                nSamtest = H_test.shape[0]
                kt = np.tile(ktr, (nSamtest, 1))
                bt = np.tile(btr, (nSamtest, 1))
                testX_temp = testX * kt + bt
                H_test = np.concatenate((H_test, testX_temp), axis=1)
            else:
                H_test = np.concatenate((H_test, testX), axis=1)

        # 将NaN值设置为0
        H_test = np.nan_to_num(H_test)

        # 计算测试数据的分类结果
        testY_temp = np.dot(H_test, beta)
        Yt_temp = np.zeros(testY.size, dtype=int)
        for i in range(testY.size):
            idx = np.argmax(testY_temp[i, :])
            Yt_temp[i] = U_trainY[idx]

        # 计算训练数据的分类精度
        train_accuracy = np.count_nonzero(Y_temp == trainY) / Nsample

        # 计算测试数据的分类精度
        test_accuracy = np.count_nonzero(Yt_temp == testY) / np.size(testY)

        return Yt_temp, train_accuracy, test_accuracy
