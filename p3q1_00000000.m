function new3q1_00000000
% P3Q1: CISC371, Practice 3, Question 1
% CISC371, Fall 2024: IRIS sepal data for species I. versicolor

    % Set the size of the ANN: Lnum neurons in hidden layer
    Lnum = 3;

    % Use part of Fisher's iris data; set k=[] to use all data
    k = [];
    [xmat, yall] = getiris(k);
    xmat = xmat(:, 3:4);

    % Select I. versicolor as the species of interest; ID number is 2
    yvec = (yall == 2);

    % Set the auxiliary data structures for functions and gradients
    global ANNDATA
    global ANNSTRUCT
    ANNDATA.lnum = Lnum;
    ANNDATA.xmat = xmat;
    ANNDATA.yvec = yvec;

    % Set the starting point: fixed weight vector
    if Lnum==3
        w0 = [-2; 1; 1; -1; 1; -1; -2; -1; -1; 0.5; 1; 0.5; 0.5];
    else
        % This setting will induce code failure
        w0 = [];
    end

    % Initialize global ANN structure: 2D inputs, 1 hidden layer of Lnum
    ANNSTRUCT = [];
    dimIn = 2;
    anninit([dimIn Lnum 1], {'sigmoid' 'heaviside'}, w0);
    
    % Set the learning rate and related parameters
    eta   = 0.5*1/size(xmat, 1);
    imax  = 5000;
    % imax = 1;
    gnorm = 1e-3;

    % Original data
    disp('   ... RAW data...');
    % Plot and pause
    figure(1);
    ph=gscatter(xmat(:,1), xmat(:,2), yvec, "rb", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. versicolor", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data', ...
        'interpreter', 'latex', 'fontSize', 14);
    pause(0.5);

    % MATLAB Builtin neural network
    disp('   ... doing NET...');
    % MATLAB wants the data transposed from the usual format
    net2layer = configure(feedforwardnet(Lnum), xmat', yvec');
    net2layer.trainParam.showWindow = 0;
    [mlnet, mltrain] = train(net2layer, xmat', yvec');
    % Network output is in [0,1] so decision threshold is 1/2
    ynet = (mlnet(xmat')>0.5);
    % Plot these results
    figure(2);
    ph=gscatter(xmat(:,1), xmat(:,2), ynet, "mc", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. versicolor", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: MATLAB network', ...
        'interpreter', 'latex', 'fontSize', 14);
    pause(0.5);

    % Hard-coded network with 1 hidden layer of Lnum neurons
    disp('   ... doing ANN response...');
    [wann fann iann] = annsteepfixed(@anngradient, ...
        w0, eta, max([1 imax]), gnorm);
    [yann,~] = annclass(wann);
    cok = 100*(1 - sum(abs(ANNDATA.yvec - yann))/numel(ANNDATA.yvec));
    % Plot and pause
    disp(sprintf('ANN (%d), W is', iann));
    disp(wann');
    fprintf('Descent: %0.1f%% correct\n', cok);
    figure(3);
    ph=gscatter(xmat(:,1), xmat(:,2), yann, "mc", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. versicolor", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: custom network', ...
        'interpreter', 'latex', 'fontSize', 14);
end

function [rvec, hrmat] = annclass(wvec)
% FUNCTION RVEC=ANNCLASS(WVEC) computes the response of a simple neural
% network that has 1 hidden layer of logistic cells and a linear output.
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC  -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -   Structure containing
%                     xmat - MxN matrix, each row is a data vector
%                     yvec - Mx1 column vector, each label is 0 or 1
%         ANNSTRUCT - Structure including
%                     responses - cell array of neuron responses
% OUTPUTS:
%         RVEC  - Mx1 vector of linear responses to data
%         HRMAT - Mx(L+1) array of hidden-layer responses to data

    % Problem size: original data, intermediate data
    global ANNDATA
    global ANNSTRUCT
    [m,n] = size(ANNDATA.xmat);
    xmat = [ANNDATA.xmat ones(m, 1)];

    [m2, n2] = size(ANNSTRUCT.wmats{2});
    annsetweights(wvec);
    % Loop through the data, accumulating responses
    rvec =zeros(m, 1);
    hrmat = [];
    for jx = 1:m
        xObs = xmat(jx, 1:(end-1));
        yVal = ANNDATA.yvec(jx);
        anneval1(xObs);
        rvec(jx) = ANNSTRUCT.responses{end};
        hrmat(jx, :) = ANNSTRUCT.responses{2};
    end
end

function anninit(layerNums, layerTypes, wvecInit)
% FUNCTION ANN=ANNINIT(LAYERNUMS,LAYERTYPES,WVEC) initializes a
% multi-layer neural network. LAYERNUMS specifies the number of neurons
% in each layer, with LAYERNUMS(1)==1 required and LAYERNUMS(end) being
% the dimension of the input observations. LAYERTYPE is either SIGMOID,
% RELU, HEAVISIDE, or else is forced to IDENTITY. Optionally, WVEC
% initializes the weight of the network.
% The structure is set in the global variable ANNSTRUCT.
%
% INPUTS:
%         LAYERNUMS - vector, number of neurons in each layer; first
%                     entry is the size of inputs, last entry is the
%                     number of responses of the network (MUST BE 1)
%         LAYERTYPE - cell array of strings, activation function per layer
%                     'sigmoid' for classical activation function
%                     'relu' for Rectified Linear Unit activation
%                     'heaviside' for unit step function
%                     'identity' for identity activation (default)
%         WVEC     -  optional, initial vaules for neuron weights
% OUTPUTS:
%         ANN      -  Structure containing
%                       lnum - number of hidden units to compute
%                       xmat - MxN matrix, each row is a data vector
%                       yvec - Mx1 column vector, each label is 0 or 1
% SIDE EFFECTS:
%         ANNSTRUCT - global structure, with fields that are cell arrays:
%                       wmats     - weight matrix for each layer
%                       actfun    - activation function for each layer
%                       responses - vectors of activations
%                     Weights for Layer 1 are all unity; response for
%                     final layer is the network response
% EXAMPLE:
%         anninit([2 2 1], {'sigmoid' 'heaviside'}) creates a network
%         with 2D inputs, one hidden layer of 2 sigmoid neurons, and
%         one output neuron with a unit-step response

    % Define and initialize the global variable
    global ANNSTRUCT
    ANNSTRUCT = [];

    % Problem size: validate the layers
    pNum = numel(layerNums);
    tNum = numel(layerTypes);
    if layerNums~=1
        disp('   ANNIT error - first layer must have 1 neuron');
        return;
    end
    if pNum~=(tNum + 1)
        disp('   ANNINIT error - must be 1 type per non-input layer');
        return;
    end

    % Set each weight to unity and fill in the activation functions
    ANNSTRUCT.wmats = {};
    ANNSTRUCT.numWeights = 0;
    for lx = 1:pNum
        level = lx;
        if lx~=1
            % Usual: weights are sized from previous layer
            thisWmat = ones((layerNums(lx-1) + 1), ...
                layerNums(lx));
            ANNSTRUCT.wmats{lx} = thisWmat;
            ANNSTRUCT.numWeights = ANNSTRUCT.numWeights ...
                + numel(thisWmat);
            switch lower(layerTypes{lx-1})
                case 'sigmoid'
                    ANNSTRUCT.actfun{lx} = 'sigmoid';
                case 'relu'
                    ANNSTRUCT.actfun{lx} = 'relu';
                case 'heaviside'
                    ANNSTRUCT.actfun{lx} = 'heaviside';
                otherwise
                    ANNSTRUCT.actfun{lx} = 'identity';
            end
        else
            % Input layer: weights are size for observations
            ANNSTRUCT.wmats{lx} = ones(layerNums(lx), 1);
            ANNSTRUCT.actfun{lx} = 'identity';
            ANNSTRUCT.responses{lx} = [];
        end
    end

    % Declare fields for feed-forward and back-propagation passes
    ANNSTRUCT.responses = [];
    ANNSTRUCT.dvec = [];
    ANNSTRUCT.backprop = [];

    % Optionally, initialize the weights as provided
    if exist('wvecInit') && ~isempty(wvecInit)
        annsetweights(wvecInit);
    end
end

function [tmin,fmin,ix]=annsteepfixed(objgradf,w0,s,imax_in,eps_in)
% [WMIN,FMIN,IX]=ANNSTEEPFIXED(OBJGRADF,W0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         WMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

    % Set convergence criteria to those supplied, if available
    if nargin >= 4 & ~isempty(imax_in)
        imax = imax_in;
    else
        imax = 50000;
    end

    if nargin >= 5 & ~isempty(eps_in)
        epsilon = eps_in;
    else
        epsilon = 1e-6;
    end

    % Initialize: search vector, objective, gradient
    tmin = w0;
    [fmin gval] = objgradf(tmin);
    ix = 0;
    while (norm(gval)>epsilon & ix<imax)

    % %
    % % STUDENT CODE GOES HERE: REPLACE "BREAK" WITH WORKING CODE
    % %
        tmin = tmin + s * -gval';
        [fmin gval] = objgradf(tmin);
        ix = ix + 1;
    end
end
