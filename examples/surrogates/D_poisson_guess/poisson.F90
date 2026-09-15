! Periodic Poisson solves with a conv-net initial guess, Fortran.
!
! The twin of poisson.c: the whole right-hand side, with a 6-cell periodic
! halo, is ONE model input (NCHW 1 x 1 x 76 x 76 -> 1 x 1 x 64 x 64), one
! poisson_guess_infer call per step; Jacobi then runs from the guess on the
! device to a residual tolerance, against a zero start and a warm start.
!
! As in poisson.c, the guess runs on the HOST and is uploaded (32 KB per
! step): the generated infer holds a whole-field model's activations as
! locals of the call, 660 KB here, beyond what a device thread can hold.
!
! Layout: the model's NCHW input is row-major with the halo'd row index
! slowest; Fortran arrays are column-major, so fp is declared fp(np_, np_)
! indexed (j, i) -- column j fastest -- to hand infer the same flat order.
program poisson
    use poisson_guess_model, only: poisson_guess_infer
    use iso_fortran_env, only: real64
    use omp_lib, only: omp_get_wtime
    implicit none
    integer, parameter :: n = 64, halo = 6, np_ = n + 2 * halo, nsteps = 20
    real(real64), parameter :: tol = 1.0e-3_real64
    integer, parameter :: max_it = 20000, check_every = 20
    real(real64), parameter :: pi = 3.14159265358979323846_real64

    real(real64), allocatable :: f(:,:), tmp(:,:), fp(:,:), phi_nn(:,:), phi_zero(:,:), phi_warm(:,:)
    integer(8) :: it_nn, it_zero, it_warm
    real(real64) :: t0, t_guess, fnorm
    integer :: s, i, j

    allocate(f(n, n), tmp(n, n), fp(np_, np_), phi_nn(n, n), phi_zero(n, n), phi_warm(n, n))
    phi_nn = 0.0_real64; phi_zero = 0.0_real64; phi_warm = 0.0_real64
    it_nn = 0; it_zero = 0; it_warm = 0; t_guess = 0.0_real64
    fnorm = sqrt(real(n * n, real64))                       ! unit rms

    ! Everything the loop touches is mapped once.
    !$omp target enter data map(alloc: f, tmp) map(to: phi_nn, phi_zero, phi_warm)
    do s = 0, nsteps - 1
        call rhs(f, s)
        !$omp target update to(f)                           ! the step's new RHS: the solver's own I/O

        ! The NN guess, on the host: periodic halo, one whole-field infer, upload.
        t0 = omp_get_wtime()
        do i = 0, np_ - 1
            do j = 0, np_ - 1
                fp(j + 1, i + 1) = f(wrap(j - halo) + 1, wrap(i - halo) + 1)
            end do
        end do
        call poisson_guess_infer(fp, phi_nn)                ! one call, the whole field
        !$omp target update to(phi_nn)                      ! the guess: 32 KB, once per step
        call zero_mean(phi_nn)
        t_guess = t_guess + (omp_get_wtime() - t0)

        it_nn = it_nn + jacobi(phi_nn, tmp, f, fnorm)
        call clear(phi_zero)
        it_zero = it_zero + jacobi(phi_zero, tmp, f, fnorm)
        it_warm = it_warm + jacobi(phi_warm, tmp, f, fnorm)
    end do
    !$omp target exit data map(delete: f, tmp, phi_nn, phi_zero, phi_warm)

    print '(I0,A,I0,A,I0,A,ES7.0,A)', n, 'x', n, ' periodic Poisson, ', nsteps, &
        ' steps of a rotating right-hand side, Jacobi to ', tol, ':'
    print '(A,F6.0)', '  iterations per step, from a zero guess              ', real(it_zero, real64) / nsteps
    print '(A,F6.0)', '  iterations per step, from the previous solution     ', real(it_warm, real64) / nsteps
    print '(A,F6.0,A,F5.1,A)', '  iterations per step, from the NN guess              ', real(it_nn, real64) / nsteps, &
        '   (guess: ', 1e3 * t_guess / nsteps, ' ms per step)'
    if (it_nn >= int(max_it, 8) * nsteps .or. it_nn >= it_zero) then
        print '(A)', 'FAIL: the NN guess did not reduce the iteration count'
        stop 1
    end if
    print '(A,F3.0,A)', 'OK: the NN guess saves ', 100.0_real64 * (1.0_real64 - real(it_nn, real64) / it_zero), &
        '% of the zero-start iterations'

contains

    pure integer function wrap(i)      ! 0-based periodic index
        !$omp declare target
        integer, intent(in) :: i
        wrap = modulo(i, n)
    end function

    ! Right-hand side at step s: six Fourier modes whose phases rotate with s (as poisson.c).
    subroutine rhs(f, s)
        real(real64), intent(out) :: f(n, n)
        integer, intent(in) :: s
        integer, parameter :: p(6) = [1, 2, -3, 4, 5, -7], q(6) = [2, -1, 3, 1, -5, 2]
        real(real64), parameter :: a(6) = [0.9_real64, -0.7_real64, 0.5_real64, 0.6_real64, -0.4_real64, 0.3_real64]
        integer :: i, j, m
        do i = 0, n - 1
            do j = 0, n - 1
                f(j + 1, i + 1) = 0.0_real64
                do m = 1, 6
                    f(j + 1, i + 1) = f(j + 1, i + 1) + a(m) * cos(2.0_real64 * pi * (p(m) * i + q(m) * j) / n &
                                                                  + 0.15_real64 * s * m)
                end do
            end do
        end do
        f = f - sum(f) / (n * n)
        f = f / sqrt(sum(f**2) / (n * n))
    end subroutine

    ! One Jacobi sweep of phi into out. f(j, i) holds poisson.c's f[i*N + j].
    subroutine sweep(phi, out, f)
        real(real64), intent(in) :: phi(n, n), f(n, n)
        real(real64), intent(out) :: out(n, n)
        integer :: i, j
        !$omp target teams distribute parallel do collapse(2)
        do i = 1, n
            do j = 1, n
                out(j, i) = 0.25_real64 * (phi(j, wrap(i - 2) + 1) + phi(j, wrap(i) + 1) &
                                           + phi(wrap(j - 2) + 1, i) + phi(wrap(j) + 1, i) - f(j, i))
            end do
        end do
    end subroutine

    real(real64) function residual(phi, f)
        real(real64), intent(in) :: phi(n, n), f(n, n)
        real(real64) :: r2, lap
        integer :: i, j
        r2 = 0.0_real64
        !$omp target teams distribute parallel do collapse(2) reduction(+: r2) private(lap)
        do i = 1, n
            do j = 1, n
                lap = phi(j, wrap(i - 2) + 1) + phi(j, wrap(i) + 1) + phi(wrap(j - 2) + 1, i) + phi(wrap(j) + 1, i) &
                      - 4.0_real64 * phi(j, i)
                r2 = r2 + (lap - f(j, i))**2
            end do
        end do
        residual = sqrt(r2)
    end function

    ! Jacobi from phi (in place, with tmp) until |lap(phi) - f| / |f| < tol; returns iterations.
    integer function jacobi(phi, tmp, f, fnorm)
        real(real64), intent(inout) :: phi(n, n), tmp(n, n)
        real(real64), intent(in) :: f(n, n), fnorm
        integer :: it
        do it = 0, max_it - 2, 2
            call sweep(phi, tmp, f)
            call sweep(tmp, phi, f)
            if (mod(it + 2, check_every) == 0) then
                if (residual(phi, f) / fnorm < tol) then
                    jacobi = it + 2
                    return
                end if
            end if
        end do
        jacobi = max_it
    end function

    subroutine zero_mean(phi)
        real(real64), intent(inout) :: phi(n, n)
        real(real64) :: m
        integer :: i, j
        m = 0.0_real64
        !$omp target teams distribute parallel do collapse(2) reduction(+: m)
        do i = 1, n
            do j = 1, n
                m = m + phi(j, i)
            end do
        end do
        m = m / (n * n)
        !$omp target teams distribute parallel do collapse(2)
        do i = 1, n
            do j = 1, n
                phi(j, i) = phi(j, i) - m
            end do
        end do
    end subroutine

    subroutine clear(phi)
        real(real64), intent(out) :: phi(n, n)
        integer :: i, j
        !$omp target teams distribute parallel do collapse(2)
        do i = 1, n
            do j = 1, n
                phi(j, i) = 0.0_real64
            end do
        end do
    end subroutine
end program
