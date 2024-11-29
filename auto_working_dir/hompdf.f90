!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   hompdf.f90 - homogenized pdf stationary points
!----------------------------------------------------------------------
!----------------------------------------------------------------------

subroutine ik01a ( x, bi0, bi1 )

! Removed di0, di1 and all k
!*****************************************************************************80
!
!! IK01A compute Bessel function I0(x), I1(x), K0(x), and K1(x).
!
!  Discussion:
!
!    This procedure computes modified Bessel functions I0(x), I1(x),
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However, 
!    they give permission to incorporate this routine into a user program 
!    provided that the copyright is acknowledged. Adapted by Doug Addy to remove
!    superfluous functionality
!
!  Modified:
!
!    01 July 2017
!
!  Author:
!
!    Shanjie Zhang, Jianming Jin
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument.
!
!    Output, real ( kind = 8 ) BI0, DI0, BI1, DI1, BK0, DK0, BK1, DK1, the
!    values of I0(x), I0'(x), I1(x), I1'(x), K0(x), K0'(x), K1(x), K1'(x).
!
  implicit none

  real ( kind = 8 ), save, dimension ( 12 ) :: a = (/ &
    0.125D+00, 7.03125D-02, &
    7.32421875D-02, 1.1215209960938D-01, &
    2.2710800170898D-01, 5.7250142097473D-01, &
    1.7277275025845D+00, 6.0740420012735D+00, &
    2.4380529699556D+01, 1.1001714026925D+02, &
    5.5133589612202D+02, 3.0380905109224D+03 /)
  real ( kind = 8 ), save, dimension ( 8 ) :: a1 = (/ &
    0.125D+00, 0.2109375D+00, &
    1.0986328125D+00, 1.1775970458984D+01, &
    2.1461706161499D+02, 5.9511522710323D+03, &
    2.3347645606175D+05, 1.2312234987631D+07 /)
  real ( kind = 8 ), save, dimension ( 12 ) :: b = (/ &
    -0.375D+00, -1.171875D-01, &
    -1.025390625D-01, -1.4419555664063D-01, &
    -2.7757644653320D-01, -6.7659258842468D-01, &
    -1.9935317337513D+00, -6.8839142681099D+00, &
    -2.7248827311269D+01, -1.2159789187654D+02, &
    -6.0384407670507D+02, -3.3022722944809D+03 /)
    
  real ( kind = 8 ) bi0
  real ( kind = 8 ) bi1
  real ( kind = 8 ) ca
  integer ( kind = 4 ) k
  integer ( kind = 4 ) k0
  real ( kind = 8 ) pi
  real ( kind = 8 ) r
  real ( kind = 8 ) x
  real ( kind = 8 ) x2
  real ( kind = 8 ) xr

  pi = 3.141592653589793D+00
  x2 = x * x

  if ( x == 0.0D+00 ) then

    bi0 = 1.0D+00
    bi1 = 0.0D+00
    return

  else if ( x <= 18.0D+00 ) then

    bi0 = 1.0D+00
    r = 1.0D+00
    do k = 1, 50
      r = 0.25D+00 * r * x2 / ( k * k )
      bi0 = bi0 + r
      if ( abs ( r / bi0 ) < 1.0D-15 ) then
        exit
      end if
    end do

    bi1 = 1.0D+00
    r = 1.0D+00
    do k = 1, 50
      r = 0.25D+00 * r * x2 / ( k * ( k + 1 ) )
      bi1 = bi1 + r
      if ( abs ( r / bi1 ) < 1.0D-15 ) then
        exit
      end if
    end do

    bi1 = 0.5D+00 * x * bi1

  else

    if ( x < 35.0D+00 ) then
      k0 = 12
    else if ( x < 50.0D+00 ) then
      k0 = 9
    else
      k0 = 7
    end if

    ca = exp ( x ) / sqrt ( 2.0D+00 * pi * x )
    bi0 = 1.0D+00
    xr = 1.0D+00 / x
    do k = 1, k0
      bi0 = bi0 + a(k) * xr ** k
    end do
    bi0 = ca * bi0
    bi1 = 1.0D+00
    do k = 1, k0
      bi1 = bi1 + b(k) * xr ** k
    end do
    bi1 = ca * bi1

  end if
  return
end

SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!--------- ----

! Evaluates the algebraic equations or ODE right hand side

! Input arguments :
!      NDIM   :   Dimension of the algebraic or ODE system 
!      U      :   State variables
!      ICP    :   Array indicating the free parameter(s)
!      PAR    :   Equation parameters

! Values to be returned :
!      F      :   Equation or ODE right hand side values

! Normally unused Jacobian arguments : IJAC, DFDU, DFDP (see manual)

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM),DFDP(NDIM,*)

  DOUBLE PRECISION x, lambda, beta, epsil, b0, b1

  x = U(1)
  
  lambda = PAR(1)
  beta = PAR(2)
  epsil = PAR(3)
  
  call ik01a (x**2*beta/2, b0, b1)
    
  F(1)= beta*(-DEXP(-beta*(x**4/4 - (lambda*x**2)/2 + x*epsil)))*(-lambda*x*b0 + epsil*b0 - x*b1 + x**3*b0)
  
END SUBROUTINE FUNC

!-----------------------------------------------------------------------
!-----------------------------------------------------------------------

SUBROUTINE STPNT(NDIM,U,PAR,T)
!--------- -----

! Input arguments :
!      NDIM   :   Dimension of the algebraic or ODE system 

! Values to be returned :
!      U      :   A starting solution vector
!      PAR    :   The corresponding equation-parameter values

! Note : For time- or space-dependent solutions this subroutine has
!        the scalar input parameter T contains the varying time or space
!        variable value.
  
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM
  DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
  DOUBLE PRECISION, INTENT(IN) :: T

! Initialize the equation parameters
  PAR(1:3) = (/ 0.0, 0.0, 0.0/)

! Initialize the solution
  U(1) = 0.0
   
END SUBROUTINE STPNT

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
!--------- ----

! Boundary Conditions

! Input arguments :
!      NDIM   :   Dimension of the ODE system 
!      PAR    :   Equation parameters
!      ICP    :   Array indicating the free parameter(s)
!      NBC    :   Number of boundary conditions
!      U0     :   State variable values at the left boundary
!      U1     :   State variable values at the right boundary

! Values to be returned :
!      FB     :   The values of the boundary condition functions 

! Normally unused Jacobian arguments : IJAC, DBC (see manual)

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
  DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
  DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
  DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)

!X FB(1)=
!X FB(2)=

END SUBROUTINE BCND

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
!--------- ----

! Integral Conditions

! Input arguments :
!      NDIM   :   Dimension of the ODE system 
!      PAR    :   Equation parameters
!      ICP    :   Array indicating the free parameter(s)
!      NINT   :   Number of integral conditions
!      U      :   Value of the vector function U at `time' t

! The following input arguments, which are normally not needed,
! correspond to the preceding point on the solution branch
!      UOLD   :   The state vector at 'time' t
!      UDOT   :   Derivative of UOLD with respect to arclength
!      UPOLD  :   Derivative of UOLD with respect to `time'

! Normally unused Jacobian arguments : IJAC, DINT

! Values to be returned :
!      FI     :   The value of the vector integrand 

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
  DOUBLE PRECISION, INTENT(IN) :: PAR(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
  DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
  DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)

!X FI(1)=

END SUBROUTINE ICND

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)
!--------- ----
!
! Defines the objective function for algebraic optimization problems
!
! Supplied variables :
!      NDIM   :   Dimension of the state equation
!      U      :   The state vector
!      ICP    :   Indices of the control parameters
!      PAR    :   The vector of control parameters
!
! Values to be returned :
!      FS      :   The value of the objective function
!
! Normally unused Jacobian argument : IJAC, DFDP

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: FS
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM),DFDP(*)

!X FS=

END SUBROUTINE FOPT

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE PVLS(NDIM,U,PAR)
!--------- ----

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

!---------------------------------------------------------------------- 
! NOTE : 
! Parameters set in this subroutine should be considered as ``solution 
! measures'' and be used for output purposes only.
! 
! They should never be used as `true'' continuation parameters. 
!
! They may, however, be added as ``over-specified parameters'' in the 
! parameter list associated with the AUTO-Constant NICP, in order to 
! print their values on the screen and in the ``p.xxx file.
!
! They may also appear in the list associated with AUTO-Constant NUZR.
!
!---------------------------------------------------------------------- 
! For algebraic problems the argument U is, as usual, the state vector.
! For differential equations the argument U represents the approximate 
! solution on the entire interval [0,1]. In this case its values must 
! be accessed indirectly by calls to GETP, as illustrated below.
!---------------------------------------------------------------------- 
!
! Set PAR(2) equal to the L2-norm of U(1)
!X PAR(2)=GETP('NRM',1,U)
!
! Set PAR(3) equal to the minimum of U(2)
!X PAR(3)=GETP('MIN',2,U)
!
! Set PAR(4) equal to the value of U(2) at the left boundary.
!X PAR(4)=GETP('BV0',2,U)
!
! Set PAR(5) equal to the pseudo-arclength step size used.
!X PAR(5)=GETP('STP',1,U)
!
!---------------------------------------------------------------------- 
! The first argument of GETP may be one of the following:
!        'NRM' (L2-norm),     'MAX' (maximum),
!        'INT' (integral),    'BV0 (left boundary value),
!        'MIN' (minimum),     'BV1' (right boundary value).
!
! Also available are
!   'STP' (Pseudo-arclength step size used).
!   'FLD' (`Fold function', which vanishes at folds).
!   'BIF' (`Bifurcation function', which vanishes at singular points).
!   'HBF' (`Hopf function'; which vanishes at Hopf points).
!   'SPB' ( Function which vanishes at secondary periodic bifurcations).
!---------------------------------------------------------------------- 


END SUBROUTINE PVLS

!----------------------------------------------------------------------
!----------------------------------------------------------------------
