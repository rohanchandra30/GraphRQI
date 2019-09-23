function [Up,Sp,Vp] = svd_update( U, S, V, A, B, force_orth )
% function [Up,Sp,Vp] = svd_update( U, S, V, A, B, force_orth )
%
% Given the SVD of
%
%   X = U*S*V'
%
% update it to be the SVD of
%
%   X + AB' = Up*Sp*Vp'
%
% Depending on A,B there may be considerable structure that could
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
  % of (I-UU')A, which is the component of "A" that is
  % orthogonal to U.
  m = U' * A;
  p = A - U*m;

  P = orth( p );
  % p may not have full rank.  If not, P will be too small.  Pad
  % with zeros.
  P = [ P zeros(size(P,1), size(p,2)-size(P,2)) ];

  Ra = P' * p;

  
  % Q is an orthogonal basis of the column-space
  % of (I-VV')b.
  n = V' * B;
  q = B - V*n;

  Q = orth( q );
  % q may not have full rank.  If not, Q will be too small.  Pad
  % with zeros.
  Q = [ Q zeros(size(Q,1), size(q,2)-size(Q,2)) ];

  Rb = Q' * q;
 
  %
  % Diagonalize K, maintaining rank
  %

  z = zeros( size(m) );
  z2 = zeros( size(m,2), size(m,2) );

  K = [ S z ; z' z2 ] + [ m; Ra ] * [ n; Rb ]';

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
  end;
  
return;