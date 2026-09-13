program unit_tests
    use iso_c_binding
    use activation_functions
    use derived_types
    use model_layers
    implicit none

    integer :: failures = 0

    call test_relu_basic()
    call test_sigmoid_midpoint()
    call test_maxpool_identity_nonsquare()
    call test_maxpool_preserves_batch()
    call test_tanh_saturates()

    if (failures > 0) then
        write(*,'(a,i0,a)') 'UNIT TESTS: ', failures, ' failure(s)'
        stop 1
    end if
    write(*,'(a)') 'UNIT TESTS: all passed'

contains

    subroutine check(cond, name)
        logical, intent(in) :: cond
        character(*), intent(in) :: name
        if (cond) then
            write(*,'(a,a)') '  ok   ', name
        else
            write(*,'(a,a)') '  FAIL ', name
            failures = failures + 1
        end if
    end subroutine

    subroutine test_relu_basic()
        real(c_double) :: x(3), y(3)
        x = [-1.0d0, 0.0d0, 2.0d0]
        y = relu(x)
        call check(all(abs(y - [0.0d0, 0.0d0, 2.0d0]) < 1.0d-12), 'relu clamps negatives')
    end subroutine

    subroutine test_sigmoid_midpoint()
        real(c_double) :: y(1)
        y = sigmoid([0.0d0])
        call check(abs(y(1) - 0.5d0) < 1.0d-12, 'sigmoid(0) == 0.5')
    end subroutine

    subroutine test_maxpool_identity_nonsquare()
        real(c_double), allocatable :: x(:,:,:,:)
        type(maxpoolLayer) :: mp
        integer :: r, c
        logical :: ok
        allocate(x(1,1,2,4))
        do r = 1, 2
            do c = 1, 4
                x(1,1,r,c) = 10.0d0*r + c
            end do
        end do
        mp%kernel_size = 1
        call max_pool(x, mp, 0, [0,0], [1,1])
        ok = all(shape(x) == [1,1,2,4])
        if (ok) ok = abs(x(1,1,1,3) - 13.0d0) < 1.0d-12 .and. &
                     abs(x(1,1,2,4) - 24.0d0) < 1.0d-12
        call check(ok, 'max_pool k=1 s=1 is identity on non-square input')
    end subroutine

    subroutine test_maxpool_preserves_batch()
        real(c_double), allocatable :: x(:,:,:,:)
        type(maxpoolLayer) :: mp
        allocate(x(2,1,2,2))
        x = 1.0d0
        x(2,:,:,:) = 2.0d0
        mp%kernel_size = 2
        call max_pool(x, mp, 0, [0,0], [1,1])
        call check(all(shape(x) == [2,1,1,1]) .and. &
                   abs(x(2,1,1,1) - 2.0d0) < 1.0d-12, &
                   'max_pool preserves the batch dimension')
    end subroutine

    subroutine test_tanh_saturates()
        real(c_double) :: y(3)
        y = tanhh([1.0d0, 720.0d0, -720.0d0])
        call check(abs(y(1) - 0.761594155955765d0) < 1.0d-12 .and. &
                   abs(y(2) - 1.0d0) < 1.0d-12 .and. &
                   abs(y(3) + 1.0d0) < 1.0d-12, &
                   'tanhh saturates instead of overflowing to NaN')
    end subroutine

end program unit_tests
