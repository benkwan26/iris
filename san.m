function san
% Code for a single articicial neuron
% IRISTEST will load Fisher's Iris data, test, and plot results

    % Use sepal measurement of Fisher's Iris data
    [xmat, yall] = getiris;
    xmat = xmat(:, 3:4);
    xaug = [xmat , ones(size(xmat, 1), 1)];

    % Select I. setosa as the species of interest; ID number is 1
    yvec = (yall == 1);

    % Place data in global variable for optimization
    global ANNDATA;
    ANNDATA = [];
    ANNDATA.xmat = xmat;
    ANNDATA.yvec = yvec;

    % IRIS data: initial hyperplane points in the wrong direction
    w0 = [ 1 ; 1 ; 1 ];

    % Use MATLAB descent method to solve the problem
    wmin = fminunc(@log1cell, w0, optimset('Display','none'));

    % Classify the data using the optimal hyperplane
    cvec = xaug*wmin >= 0;
    cok = 100*(1 - sum(ANNDATA.yvec - cvec)/numel(ANNDATA.yvec));

    % Plot the data
    clf;
    ph=gscatter(xmat(:,1), xmat(:,2), yvec, "rb", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    % Show the separating hyperplane as a level curve at zero
    axv = axis;
    gden = 100;
    [xg,yg] = meshgrid(linspace(axv(1), axv(2), gden), ...
        linspace(axv(3), axv(4), gden));
    fimp =@(x1,x2) wmin(1)*x1 + wmin(2)*x2 + wmin(3);
    hold on;
    contour(xg,yg,fimp(xg,yg), [0 0], 'color', 'k', 'LineWidth', 2);
    hold off;
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title(sprintf('IRIS data: %d\\%% correct', cok), ...
        'Interpreter', 'latex', 'FontSize', 14);
end

function [fval,g1form] = log1cell(wvec)
% [FVAL,G1FORM]=LOGCELL(WVEC) computes the objective function and the
% gradient 1-form for weights WVEC in a cell with a logistic activation.
% Data are in the global variable ANNDATA. The objective is the
% sum of squared residual errors.
% Here, N is the number of independent entries of the augmented weight
% vector
%
% INPUTS:
%         WVEC    - Nx1 augmented weight vector
% OUTPUTS:
%         FVAL    - scalar, objective value for this weight vector
%         G1FORM  - 1xN gradient as a 1-form
% NEEDS:
%         ANNDATA - global variable with observations and labels

    % Declare the global variable and find sizes
    global ANNDATA
    [M,N] = size(ANNDATA.xmat);
    L = size(ANNDATA.yvec, 2);
    % Append 1's to the observations
    xmat = [ANNDATA.xmat , ones(M, 1)];

    % Anonymous functions for the logistic activation and derivative
    phi_log =@(u) 1./(1 + exp(-u));
    psi_log =@(u) phi_log(u).*(1 - phi_log(u));

    % Find the inner product vector
    uvec = xmat*wvec;

    % Find the score, or activation
    zvec = phi_log(uvec);

    % Compute the residual-error vector and the objective value
    rvec = ANNDATA.yvec(:,1) - zvec;
    fval = 0.5*rvec'*rvec;
    
    % Loop to compute the gradients
    % Gradient matrix has a 1-form for each observation
    gmat = zeros(size(xmat));
    for ix = 1:size(xmat, 1)
        gmat(ix,:) = -rvec(ix)*psi_log(uvec(ix))*xmat(ix,:);
    end
    % Gradient is the sum of observation gradients
    g1form = sum(gmat, 1);
end
