program weights_format_tests
    ! Weights-format tests on a one-layer Gemm model, one mode per process.
    !   write     model, .bin, extra.bin, legacy.TXT, onnxWeights.txt
    !   bin       explicit .bin path
    !   txt       uppercase .TXT path with trailing blanks
    !   extra     oversized .bin, must be rejected
    !   fallback  no arguments, no .bin
    use iso_c_binding
    use reader
    implicit none
    character(len=16) :: mode
    real(c_double) :: v(8)
    integer :: k
    logical :: exists

    v = [(0.1d0*k - 0.35d0, k = 1, 8)]
    call get_command_argument(1, mode)
    select case (trim(mode))
    case ('write')
        call write_files()
    case ('bin')
        call initialize("onnxModel.txt"//c_null_char, "onnxWeights.bin"//c_null_char)
        call check_values('explicit .bin path loads the weights')
    case ('txt')
        call initialize("onnxModel.txt"//c_null_char, "legacy.TXT   "//c_null_char)
        call check_values('uppercase .TXT path with trailing blanks loads the same weights as the .bin')
    case ('extra')
        call initialize("onnxModel.txt"//c_null_char, "extra.bin"//c_null_char)
        write(*,'(a)') '  FAIL a .bin with extra bytes was accepted'
    case ('fallback')
        inquire(file='onnxWeights.bin', exist=exists)
        if (exists) error stop 'fallback mode needs onnxWeights.bin to be absent'
        call initialize()
        call check_values('no arguments and no onnxWeights.bin falls back to onnxWeights.txt')
    case default
        error stop 'unknown mode'
    end select

contains

    subroutine write_files()
        integer :: u
        open(newunit=u, file='onnxModel.txt', status='replace', action='write')
        write(u,'(a)') '1'
        write(u,'(a)') 'Gemm'
        write(u,'(a)') '2 3'
        write(u,'(a)') '2'
        close(u)
        open(newunit=u, file='onnxWeights.bin', status='replace', action='write', &
            access='stream', form='unformatted')
        write(u) v
        close(u)
        open(newunit=u, file='extra.bin', status='replace', action='write', &
            access='stream', form='unformatted')
        write(u) v
        write(u) 1.0d0
        close(u)
        call write_text('legacy.TXT')
        call write_text('onnxWeights.txt')
    end subroutine

    subroutine write_text(path)
        ! one value per line is valid legacy text for any model
        character(*), intent(in) :: path
        integer :: u
        open(newunit=u, file=path, status='replace', action='write')
        do k = 1, size(v)
            write(u,'(es26.17e3)') v(k)
        end do
        close(u)
    end subroutine

    subroutine check_values(name)
        character(*), intent(in) :: name
        logical :: ok
        ok = size(linLayers) == 1
        if (ok) ok = all(shape(linLayers(1)%weights) == [2, 3]) .and. size(linLayers(1)%biases) == 2
        if (ok) ok = all(linLayers(1)%weights == reshape(v(1:6), [2, 3])) .and. &
                     all(linLayers(1)%biases == v(7:8))
        if (ok) then
            write(*,'(a,a)') '  ok   ', name
        else
            write(*,'(a,a)') '  FAIL ', name
            error stop 2
        end if
    end subroutine

end program
