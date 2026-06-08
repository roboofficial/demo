
%% EXPERIMENT 4: Path Planning — A*, Dijkstra, D* Lite
%  DIAT Robotics Lab | No extra toolboxes needed | Press F5 to run
clear; clc; close all;

%% ================================================================
%% 1. CREATE MAP  (40x40 grid, each cell = 0.5 m)
%% ================================================================
ROWS = 40; COLS = 40;
map = zeros(ROWS, COLS);        % 0=free, 1=obstacle

map(10,  5:30)    = 1;          % horizontal wall
map(5:25, 20)     = 1;          % vertical wall
map(13:16, 20)    = 0;          % gap in wall
map(25:35, 8:12)  = 1;          % L-block
map(30:35, 12:18) = 1;
map(18:22, 28:32) = 1;          % square block
map(1,:)=1; map(end,:)=1;       % border
map(:,1)=1; map(:,end)=1;

START = [3,  3];
GOAL  = [37, 37];

%% ================================================================
%% 2. RUN ALGORITHMS
%% ================================================================
tic; [path_as, exp_as] = fn_astar(map, START, GOAL);     t_as = toc;
tic; [path_dj, exp_dj] = fn_dijkstra(map, START, GOAL);  t_dj = toc;
tic; [path_ds, exp_ds] = fn_dstar(map, START, GOAL);      t_ds = toc;

%% ================================================================
%% 3. SIDE-BY-SIDE MAPS
%% ================================================================
figure('Name','Path Planning','Position',[50 50 1400 480]);
names  = {'A*','Dijkstra','D* Lite'};
paths  = {path_as, path_dj, path_ds};
exps   = {exp_as,  exp_dj,  exp_ds};
times  = [t_as, t_dj, t_ds];
colors = {[0 0.45 0.74],[0.85 0.33 0.10],[0.47 0.67 0.19]};
for i = 1:3
    subplot(1,3,i);
    plot_map(map, paths{i}, exps{i}, START, GOAL, colors{i});
    title(sprintf('%s\nTime:%.4fs  Nodes:%d  Len:%.1f', ...
        names{i}, times(i), size(exps{i},1), plen(paths{i})), 'FontSize',9);
end

%% ================================================================
%% 4. BAR CHARTS
%% ================================================================
figure('Name','Performance','Position',[50 600 800 300]);
dat  = [times*1000;
        size(exp_as,1) size(exp_dj,1) size(exp_ds,1);
        plen(path_as)  plen(path_dj)  plen(path_ds)];
ylbs = {'Time (ms)','Nodes Expanded','Path Length (cells)'};
ttls = {'Planning Time','Nodes Expanded','Path Length'};
bc   = {[0.3 0.6 0.9],[0.9 0.5 0.3],[0.4 0.8 0.4]};
for i = 1:3
    subplot(1,3,i);
    b = bar(dat(i,:)); grid on;
    for j=1:3, b.FaceColor='flat'; b.CData(j,:)=bc{i}; end
    set(gca,'XTickLabel',names); ylabel(ylbs{i}); title(ttls{i});
end

%% ================================================================
%% 5. CONSOLE TABLE
%% ================================================================
fprintf('\n%s\n',repmat('=',1,55));
fprintf('%-10s  %-13s  %-15s  %-10s\n','Algorithm','Time(ms)','Nodes','PathLen');
fprintf('%s\n',repmat('-',1,55));
for i = 1:3
    fprintf('%-10s  %-13.3f  %-15d  %-10.1f\n', ...
        names{i}, times(i)*1000, size(exps{i},1), plen(paths{i}));
end
fprintf('%s\n\n',repmat('=',1,55));


%% ================================================================
%% ---- A* ----
%% ================================================================
function [path, expanded] = fn_astar(map, S, G)
    [R,C] = size(map);
    INF   = 1e9;
    gcost = INF*ones(R,C);  gcost(S(1),S(2)) = 0;
    par   = zeros(R,C,2);
    vis   = false(R,C);
    % open: Nx3  [row col f]
    openL = [S(1) S(2) h(S,G)];
    expanded = zeros(0,2);

    while ~isempty(openL)
        [~,idx] = min(openL(:,3));
        cur = openL(idx,1:2);
        openL(idx,:) = [];
        if vis(cur(1),cur(2)), continue; end
        vis(cur(1),cur(2)) = true;
        expanded(end+1,:) = cur;
        if cur(1)==G(1) && cur(2)==G(2)
            path = backtrack(par,G,S); return;
        end
        for nb = nbrs(cur,R,C)
            n = nb{1};
            if map(n(1),n(2)) || vis(n(1),n(2)), continue; end
            ng = gcost(cur(1),cur(2)) + dc(cur,n);
            if ng < gcost(n(1),n(2))
                gcost(n(1),n(2))  = ng;
                par(n(1),n(2),:) = cur;
                f   = ng + h(n,G);
                row = find(openL(:,1)==n(1) & openL(:,2)==n(2),1);
                if isempty(row), openL(end+1,:) = [n(1) n(2) f];
                else,            openL(row,3)   = f; end
            end
        end
    end
    path = [];
end

%% ---- Dijkstra ----
function [path, expanded] = fn_dijkstra(map, S, G)
    [R,C] = size(map);
    INF   = 1e9;
    gcost = INF*ones(R,C);  gcost(S(1),S(2)) = 0;
    par   = zeros(R,C,2);
    vis   = false(R,C);
    openL = [S(1) S(2) 0];
    expanded = zeros(0,2);

    while ~isempty(openL)
        [~,idx] = min(openL(:,3));
        cur = openL(idx,1:2);
        openL(idx,:) = [];
        if vis(cur(1),cur(2)), continue; end
        vis(cur(1),cur(2)) = true;
        expanded(end+1,:) = cur;
        if cur(1)==G(1) && cur(2)==G(2)
            path = backtrack(par,G,S); return;
        end
        for nb = nbrs(cur,R,C)
            n = nb{1};
            if map(n(1),n(2)) || vis(n(1),n(2)), continue; end
            ng = gcost(cur(1),cur(2)) + dc(cur,n);
            if ng < gcost(n(1),n(2))
                gcost(n(1),n(2))  = ng;
                par(n(1),n(2),:) = cur;
                row = find(openL(:,1)==n(1) & openL(:,2)==n(2),1);
                if isempty(row), openL(end+1,:) = [n(1) n(2) ng];
                else,            openL(row,3)   = ng; end
            end
        end
    end
    path = [];
end

%% ---- D* Lite ----
%  Backward search from GOAL to START.
%  On a static map it finds the same optimal path as A*.
function [path, expanded] = fn_dstar(map, S, G)
    [R,C] = size(map);
    INF  = 1e9;
    g    = INF*ones(R,C);
    rhs  = INF*ones(R,C);
    rhs(G(1),G(2)) = 0;
    km   = 0;

    % open list: Nx4  [row col k1 k2]
    openL    = [G(1) G(2) h(G,S) 0];
    expanded = zeros(0,2);

    for itr = 1:80000
        if isempty(openL), break; end

        % find min-key row  (lex order on k1 then k2)
        [~,ord] = sortrows(openL, [3 4]);
        best    = ord(1);
        u       = openL(best, 1:2);
        k_old   = openL(best, 3:4);
        openL(best,:) = [];

        expanded(end+1,:) = u;

        k_new = dkey(u, S, g, rhs, km);
        if k_old(1) < k_new(1) || ...
           (k_old(1)==k_new(1) && k_old(2) < k_new(2))
            % stale entry — reinsert with fresh key
            openL = upsert(openL, u, k_new);
            continue;
        end

        gu = g(u(1),u(2));
        ru = rhs(u(1),u(2));

        if gu > ru
            % overconsistent: accept
            g(u(1),u(2)) = ru;
            for nb = nbrs(u,R,C)
                s = nb{1};
                if map(s(1),s(2)), continue; end
                candidate = g(u(1),u(2)) + dc(s,u);
                if candidate < rhs(s(1),s(2))
                    rhs(s(1),s(2)) = candidate;
                end
                if rhs(s(1),s(2)) ~= g(s(1),s(2))
                    openL = upsert(openL, s, dkey(s,S,g,rhs,km));
                end
            end
        else
            % underconsistent: raise
            g(u(1),u(2)) = INF;
            % re-evaluate u itself and its neighbours
            todo = nbrs(u,R,C);
            todo{end+1} = u;
            for ni = 1:numel(todo)
                s = todo{ni};
                if iscell(s), s = s{1}; end
                if map(s(1),s(2)), continue; end
                if ~(s(1)==G(1) && s(2)==G(2))
                    best_c = INF;
                    for nb2 = nbrs(s,R,C)
                        sp = nb2{1};
                        if map(sp(1),sp(2)), continue; end
                        v = g(sp(1),sp(2)) + dc(s,sp);
                        if v < best_c, best_c = v; end
                    end
                    rhs(s(1),s(2)) = best_c;
                end
                if rhs(s(1),s(2)) ~= g(s(1),s(2))
                    openL = upsert(openL, s, dkey(s,S,g,rhs,km));
                else
                    % remove consistent nodes
                    mask  = ~(openL(:,1)==s(1) & openL(:,2)==s(2));
                    openL = openL(mask,:);
                end
            end
        end

        % terminate when start is settled
        if g(S(1),S(2))==rhs(S(1),S(2)) && isempty(openL)
            break;
        end
        if g(S(1),S(2))==rhs(S(1),S(2))
            k_s = dkey(S,S,g,rhs,km);
            if isempty(openL) || ...
               (k_s(1) >= openL(1,3) && k_s(2) >= openL(1,4))
                break;
            end
        end
    end

    path = greedy_path(g, map, S, G, R, C);
end

%% ---- D* Lite helpers ----
function k = dkey(u, S, g, rhs, km)
    mn = min(g(u(1),u(2)), rhs(u(1),u(2)));
    k  = [mn + h(u,S) + km,  mn];
end

function openL = upsert(openL, u, k)
    row = find(openL(:,1)==u(1) & openL(:,2)==u(2), 1);
    if isempty(row)
        openL(end+1,:) = [u(1) u(2) k(1) k(2)];
    else
        openL(row, 3:4) = k;
    end
end

function path = greedy_path(g, map, S, G, R, C)
    path = S;  cur = S;
    max_steps = R*C;
    step = 0;
    while ~(cur(1)==G(1) && cur(2)==G(2)) && step < max_steps
        step = step + 1;
        best = [];  bg = 1e9;
        for nb = nbrs(cur,R,C)
            n = nb{1};
            if map(n(1),n(2)), continue; end
            if g(n(1),n(2)) < bg
                bg = g(n(1),n(2));  best = n;
            end
        end
        if isempty(best) || bg >= 1e8, path = []; return; end
        path(end+1,:) = best;
        cur = best;
    end
end

%% ---- Common helpers ----
function hv = h(a, b)
    dx = abs(a(1)-b(1));  dy = abs(a(2)-b(2));
    hv = max(dx,dy) + (sqrt(2)-1)*min(dx,dy);
end

function c = dc(a, b)
    if a(1)==b(1) || a(2)==b(2),  c = 1;  else,  c = sqrt(2);  end
end

function nbs = nbrs(node, R, C)
    D = [-1 0;1 0;0 -1;0 1;-1 -1;-1 1;1 -1;1 1];
    nbs = {};
    for i = 1:8
        r = node(1)+D(i,1);  c = node(2)+D(i,2);
        if r>=1 && r<=R && c>=1 && c<=C
            nbs{end+1} = [r c]; %#ok<AGROW>
        end
    end
end

function path = backtrack(par, G, S)
    path = G;  cur = G;
    max_steps = 10000;
    step = 0;
    while ~(cur(1)==S(1) && cur(2)==S(2)) && step < max_steps
        step = step + 1;
        p = squeeze(par(cur(1),cur(2),:))';
        if all(p==0), path=[]; return; end
        path = [p; path]; %#ok<AGROW>
        cur  = p;
    end
end

function L = plen(path)
    if isempty(path) || size(path,1)<2,  L=0;  return;  end
    L = sum(sqrt(sum(diff(double(path)).^2, 2)));
end

%% ---- Visualisation ----
function plot_map(map, path, expanded, S, G, col)
    [R,C2] = size(map);
    img = ones(R,C2,3)*0.94;
    for r = 1:R
        for c = 1:C2
            if map(r,c),  img(r,c,:) = 0.18;  end
        end
    end
    imshow(img,'InitialMagnification','fit');  hold on;
    if ~isempty(expanded) && size(expanded,1) > 0
        plot(expanded(:,2), expanded(:,1), '.', ...
             'Color', [col, 0.2], 'MarkerSize', 3);
    end
    if ~isempty(path) && size(path,1) > 1
        plot(path(:,2), path(:,1), '-', 'Color', col, 'LineWidth', 2.5);
    end
    plot(S(2),S(1),'go','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','g');
    plot(G(2),G(1),'r*','MarkerSize',12,'LineWidth',2);
    legend('','Explored','Path','Start','Goal','Location','northwest','FontSize',7);
    axis tight; box on;
end
