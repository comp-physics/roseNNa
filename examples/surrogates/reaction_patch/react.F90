! 2-D FitzHugh-Nagumo with a learned time-stepper; the twin of react.c. The
! batched call goes to the C archive through the module's bind(C) routes:
! stepper_init_dev (the archive's init), stepper_infer_batch_dev with c_loc of
! the arrays inside `target data use_device_addr`, and stepper_sync_dev.
program react
    use stepper_model, only: stepper_init_dev, stepper_infer_batch_dev, stepper_sync_dev
    use iso_fortran_env, only: real64
    use iso_c_binding, only: c_loc, c_null_ptr, c_null_char
    use omp_lib, only: omp_get_wtime
    implicit none
#ifndef NX
#define NX 256
#endif
    integer, parameter :: nx_ = NX, ncell = nx_ * nx_
    real(real64), parameter :: du = 1.0_real64, dv = 0.05_real64
    real(real64), parameter :: pa = 0.7_real64, pb = 0.8_real64, eps = 0.08_real64, dt = 0.1_real64
    integer, parameter :: k = 10, nbig = 100, nfeat = 18
    real(real64), parameter :: tol = 0.05_real64
    real(real64), parameter :: pi = 3.14159265358979323846_real64

    real(real64), allocatable, target :: u(:,:), v(:,:), ur(:,:), vr(:,:), ut(:,:), vt(:,:)
    real(real64), allocatable, target :: feat(:,:), out(:,:)
    real(real64) :: t0, t_ref, t_nn, err
    integer :: status

    status = stepper_init_dev('stepper.rwt' // c_null_char)     ! the one upload
    if (status /= 0) then
        print '(A,I0)', 'stepper_init failed with status ', status
        stop 2
    end if

    allocate(u(nx_, nx_), v(nx_, nx_), ur(nx_, nx_), vr(nx_, nx_), ut(nx_, nx_), vt(nx_, nx_))
    allocate(feat(nfeat, ncell), out(2, ncell))
    call initial_fields(u, v)
    ur = u; vr = v

    !$omp target enter data map(to: ur, vr) map(alloc: ut, vt)
    t0 = omp_get_wtime()
    call reference(ur, vr, ut, vt)
    t_ref = omp_get_wtime() - t0
    !$omp target exit data map(from: ur, vr) map(delete: ut, vt)

    !$omp target enter data map(to: u, v) map(alloc: feat, out)
    t0 = omp_get_wtime()
    call surrogate(u, v, feat, out, status)
    t_nn = omp_get_wtime() - t0
    !$omp target exit data map(from: u, v) map(delete: feat, out)
    if (status /= 0) then
        print '(A,I0)', 'stepper_infer_batch failed with status ', status
        stop 3
    end if

    err = sqrt((sum((u - ur)**2) + sum((v - vr)**2)) / (sum(ur**2) + sum(vr**2)))
    print '(I0,A,I0,A,I0,A,I0,A)', nx_, 'x', nx_, ' grid, ', nbig, ' surrogate steps of ', k, ' fine steps each:'
    print '(A,F8.1,A,F6.2,A)', '  fine reference  ', 1e3 * t_ref, ' ms  (', 1e3 * t_ref / nbig, ' ms per big step)'
    print '(A,F8.1,A,F6.2,A)', '  surrogate       ', 1e3 * t_nn, ' ms  (', 1e3 * t_nn / nbig, ' ms per big step)'
    print '(A,ES10.3)', '  relative L2 error of the surrogate: ', err
    if (.not. (err < tol)) then
        print '(A,F4.2)', 'FAIL: error above ', tol
        stop 1
    end if
    print '(A)', 'OK'

contains

    pure integer function wrap(i)
        !$omp declare target
        integer, intent(in) :: i
        wrap = modulo(i - 1, nx_) + 1
    end function

    subroutine fine_step(u, v, un, vn)
        real(real64), intent(in) :: u(nx_, nx_), v(nx_, nx_)
        real(real64), intent(out) :: un(nx_, nx_), vn(nx_, nx_)
        real(real64) :: lu, lv
        integer :: i, j
        !$omp target teams distribute parallel do collapse(2) private(lu, lv)
        do j = 1, nx_
            do i = 1, nx_
                lu = u(wrap(i - 1), j) + u(wrap(i + 1), j) + u(i, wrap(j - 1)) + u(i, wrap(j + 1)) - 4.0_real64 * u(i, j)
                lv = v(wrap(i - 1), j) + v(wrap(i + 1), j) + v(i, wrap(j - 1)) + v(i, wrap(j + 1)) - 4.0_real64 * v(i, j)
                un(i, j) = u(i, j) + dt * (du * lu + u(i, j) - u(i, j)**3 / 3.0_real64 - v(i, j))
                vn(i, j) = v(i, j) + dt * (dv * lv + eps * (u(i, j) + pa - pb * v(i, j)))
            end do
        end do
    end subroutine

    subroutine reference(ur, vr, ut, vt)
        real(real64), intent(inout) :: ur(nx_, nx_), vr(nx_, nx_), ut(nx_, nx_), vt(nx_, nx_)
        integer :: s
        do s = 1, nbig * k / 2
            call fine_step(ur, vr, ut, vt)
            call fine_step(ut, vt, ur, vr)
        end do
    end subroutine

    ! Patch order as in train.py: u's 3x3 row-major (first index slowest, as
    ! react.c's i), then v's. Cell c is react.c's i*NX + j.
    subroutine gather(u, v, feat)
        real(real64), intent(in) :: u(nx_, nx_), v(nx_, nx_)
        real(real64), intent(out) :: feat(nfeat, ncell)
        integer :: i, j, di, dj, c
        !$omp target teams distribute parallel do collapse(2) private(di, dj, c)
        do j = 1, nx_
            do i = 1, nx_
                c = (i - 1) * nx_ + j
                do di = -1, 1
                    do dj = -1, 1
                        feat((di + 1) * 3 + (dj + 1) + 1, c) = u(wrap(i + di), wrap(j + dj))
                        feat(9 + (di + 1) * 3 + (dj + 1) + 1, c) = v(wrap(i + di), wrap(j + dj))
                    end do
                end do
            end do
        end do
    end subroutine

    subroutine scatter(out, u, v)
        real(real64), intent(in) :: out(2, ncell)
        real(real64), intent(out) :: u(nx_, nx_), v(nx_, nx_)
        integer :: i, j
        !$omp target teams distribute parallel do collapse(2)
        do j = 1, nx_
            do i = 1, nx_
                u(i, j) = out(1, (i - 1) * nx_ + j)
                v(i, j) = out(2, (i - 1) * nx_ + j)
            end do
        end do
    end subroutine

    subroutine surrogate(u, v, feat, out, status)
        real(real64), intent(inout), target :: u(nx_, nx_), v(nx_, nx_), feat(nfeat, ncell), out(2, ncell)
        integer, intent(out) :: status
        integer :: s
        status = 0
        do s = 1, nbig
            call gather(u, v, feat)
            !$omp target data use_device_addr(feat, out)
            status = stepper_infer_batch_dev(ncell, c_loc(feat), c_loc(out), c_null_ptr)
            !$omp end target data
            if (status /= 0) return
            status = stepper_sync_dev(c_null_ptr)
            if (status /= 0) return
            call scatter(out, u, v)
        end do
    end subroutine

    pure real(real64) function hash01(a, b)
        integer, intent(in) :: a, b
        hash01 = real(mod((int(a, 8) * 40503_8 + int(b, 8)) * 2654435761_8, 4294967296_8), real64) &
                 / 4294967296.0_real64
    end function

    subroutine initial_fields(u, v)
        real(real64), intent(out) :: u(nx_, nx_), v(nx_, nx_)
        real(real64) :: x, y, su, sv, ph
        integer :: i, j, m, p, q
        do j = 1, nx_
            do i = 1, nx_
                x = real(i - 1, real64) / nx_; y = real(j - 1, real64) / nx_
                su = 0.0_real64; sv = 0.0_real64
                do m = 0, 3
                    p = 1 + int(3 * hash01(7, m)); q = 1 + int(3 * hash01(8, m))
                    ph = 2.0_real64 * pi * (p * x + q * y)
                    su = su + (2.0_real64 * hash01(9, m) - 1.0_real64) * sin(ph + 2.0_real64 * pi * hash01(10, m))
                    sv = sv + (hash01(11, m) - 0.5_real64) * sin(ph + 2.0_real64 * pi * hash01(12, m))
                end do
                u(i, j) = 2.0_real64 * tanh(su)
                v(i, j) = 0.5_real64 * tanh(sv)
            end do
        end do
    end subroutine
end program
