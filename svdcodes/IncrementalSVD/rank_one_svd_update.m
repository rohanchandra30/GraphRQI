function [Up,Sp,Vp] = rank_one_svd_update( U, S, V, a, b, force_orth )
% function [Up,Sp,Vp] = rank_one_svd_update( U, S, V, a, b, force_orth )
%
% Given the SVD of
%
%   X = U*S*V'
%
% update it to be the SVD of
%
%   X + ab' = Up*Sp*Vp'
%
% that is, implement a rank-one update to the SVD of X.
%
% Depending on a,b there may be considerable structure that could
% be exploited, but which this code does not.
%
% The subspace rotations involved may not preserve orthogonality due
% to numerical round-off errors.  To compensate, you can set the
% "force_orth" flag, which will force orthogonality via a QR plus
% another SVD.  In a long loop, you may want to force orthogonality
% every so often.
%
% See Matthew Brand, "Fast low-rank modifications of the thin
% singular value decomposition".
%
% D. Wingate 8/17/2007
%

  current_rank = size( U, 2 );

  % P is an orthogonal basis of the column-space
  % of (I-UU')a, which is the component of "a" that is
  % orthogonal to U.
  m = U' * a;
  p = a - U*m;
  Ra = sqrt(p'*p);
  P = (1/Ra)*p;

  % XXX this has problems if a is already in the column space of U!
  % I don't know what to do in that case.
  if ( Ra < 1e-13 )
    fprintf('------> Whoa! No orthogonal component of m!\n');
  end
  
  % Q is an orthogonal basis of the column-space
  % of (I-VV')b.
  n = V' * b;
  q = b - V*n;
  Rb = sqrt(q'*q);
  Q = (1/Rb)*q;

  if ( Rb < 1e-13 )
    fprintf('------> Whoa! No orthogonal component of n!\n');
%   [tUp,tSp,tVp] = svds( K, current_rank );
% 
%   %
%   % Now update our matrices!
%   %
%   
%   Sp = tSp;
% 
%   Up = [ U P ] * tUp;
%   Vp = [ V Q ] * tVp;
% 
%   % The above rotations may not preserve orthogonality, so we explicitly
%   % deal with that via a QR plus another SVD.  In a long loop, you may
%   % want to force orthogonality every so often.
% 
%   if ( force_orth )
%     [UQ,UR] = qr( Up, 0 );
%     [VQ,VR] = qr( Vp, 0 );
%     [tUp,tSp,tVp] = svds( UR * Sp * VR', current_rank );
%     Up = UQ * tUp;
%     Vp = VQ * tVp;
%     Sp = tSp;
%   end
%   
% return
  end
  
  %
  % Diagonalize K, maintaining rank
  %

  % XXX note that this diagonal-plus-rank-one, so we should be able
  % to take advantage of the structure!
  z = zeros( size(m) );

  K = [ S z ; z' 0 ] + [ m; Ra ]*[ n; Rb ]';

  [tUp,tSp,tVp] = svds( K, current_rank );

  %
  % Now update our matrices!
  %
  
  Sp = tSp;

  Up = [ U P ] * tUp;
  Vp = [ V Q ] * tVp;

  % The above rotations may not preserve orthogonality, so we explicitly
  % deal with that via a QR plus another SVD.  In a long loop, you may
  % want to force orthogonality every so often.

  if ( force_orth )
    [UQ,UR] = qr( Up, 0 );
    [VQ,VR] = qr( Vp, 0 );
    [tUp,tSp,tVp] = svds( UR * Sp * VR', current_rank );
    Up = UQ * tUp;
    Vp = VQ * tVp;
    Sp = tSp;
  end
  
return
