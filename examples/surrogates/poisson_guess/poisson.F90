! Periodic Poisson solves with a conv-net initial guess; the twin of poisson.c.
! The guess runs on the device through the archive's infer_one (bind(C)
! route poisson_guess_infer_one_dev). fp is fp(np_, np_) indexed (j, i): the
! model's NCHW input is row-major with the row index slowest, and Fortran is
! column-major.
program poisson
    use poisson_guess_model, only: poisson_guess_infer_one_dev, poisson_guess_sync_dev
    use iso_fortran_env, only: real64
    use iso_c_binding, only: c_loc, c_null_ptr
    use omp_lib, only: omp_get_wtime
    implicit none
    integer, parameter :: n = 64, halo = 6, np_ = n + 2 * halo, nsteps = 20
    real(real64), parameter :: tol = 1.0e-3_real64
    integer, parameter :: max_it = 20000, check_every = 20
    real(real64), parameter :: pi = 3.14159265358979323846_real64

    real(real64), allocatable, target :: f(:,:), tmp(:,:), fp(:,:), phi_nn(:,:), phi_zero(:,:), phi_warm(:,:)
    integer(8) :: it_nn, it_zero, it_warm
    real(real64) :: t0, t_guess, fnorm
    integer :: s, status

    allocate(f(n, n), tmp(n, n), fp(np_, np_), phi_nn(n, n), phi_zero(n, n), phi_warm(n, n))
    phi_nn = 0.0_real64; phi_zero = 0.0_real64; phi_warm = 0.0_real64
    it_nn = 0; it_zero = 0; it_warm = 0; t_guess = 0.0_real64
    fnorm = sqrt(real(n * n, real64))

    !$omp target enter data map(alloc: f, tmp, fp) map(to: phi_nn, phi_zero, phi_warm)
    do s = 0, nsteps - 1
        call rhs(f, s)
        !$omp target update to(f)

        t0 = omp_get_wtime()
        call wrap_halo(f, fp)
        !$omp target data use_device_addr(fp, phi_nn)
        status = poisson_guess_infer_one_dev(c_loc(fp), c_loc(phi_nn), c_null_ptr)   ! on the device
        !$omp end target data
        if (status == 0) status = poisson_guess_sync_dev(c_null_ptr)
        if (status /= 0) then
            print '(A,I0)', 'infer_one failed with status ', status
            stop 3
        end if
        call zero_mean(phi_nn)
        t_guess = t_guess + (omp_get_wtime() - t0)

        it_nn = it_nn + jacobi(phi_nn, tmp, f, fnorm)
        call clear(phi_zero)
        it_zero = it_zero + jacobi(phi_zero, tmp, f, fnorm)
        it_warm = it_warm + jacobi(phi_warm, tmp, f, fnorm)
    end do
    !$omp target exit data map(delete: f, tmp, fp, phi_nn, phi_zero, phi_warm)

    print '(I0,A,I0,A,I0,A,ES7.0,A)', n, 'x', n, ' periodic Poisson, ', nsteps, &
        ' steps of a rotating right-hand side, Jacobi to ', tol, ':'
    print '(A,F6.0)', '  iterations per step from zero               ', real(it_zero, real64) / nsteps
    print '(A,F6.0)', '  iterations per step from the last solution  ', real(it_warm, real64) / nsteps
    print '(A,F6.0,A,F5.1,A)', '  iterations per step from the NN guess       ', real(it_nn, real64) / nsteps, &
        '   (guess: ', 1e3 * t_guess / nsteps, ' ms per step)'
    if (it_nn >= int(max_it, 8) * nsteps .or. it_nn >= it_zero) then
        print '(A)', 'FAIL: NN guess did not help'
        stop 1
    end if
    print '(A,F3.0,A)', 'OK: NN guess saves ', 100.0_real64 * (1.0_real64 - real(it_nn, real64) / it_zero), &
        '% of the zero-start iterations'

contains

    pure integer function wrap(i)      ! 0-based periodic index
        !$omp declare target
        integer, intent(in) :: i
        wrap = modulo(i, n)
    end function

    ! Six Fourier modes whose phases advance with the step; mean zero, unit rms.
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

    subroutine wrap_halo(f, fp)                  ! periodic halo, on the device
        real(real64), intent(in) :: f(n, n)
        real(real64), intent(out) :: fp(np_, np_)
        integer :: i, j
        !$omp target teams distribute parallel do collapse(2)
        do i = 0, np_ - 1
            do j = 0, np_ - 1
                fp(j + 1, i + 1) = f(wrap(j - halo) + 1, wrap(i - halo) + 1)
            end do
        end do
    end subroutine

    ! One Jacobi sweep; f(j, i) holds poisson.c's f[i*N + j].
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

    ! Jacobi in place until |lap(phi) - f| / |f| < tol; returns iterations.
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
