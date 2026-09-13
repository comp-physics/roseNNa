module reader

    USE derived_types
    USE activation_functions
    use iso_c_binding
    use iso_fortran_env, only: error_unit

    implicit none


    TYPE(linLayer), ALLOCATABLE, DIMENSION(:) :: linLayers
    TYPE(lstmLayer), ALLOCATABLE, DIMENSION(:) :: lstmLayers
    TYPE(convLayer), ALLOCATABLE, DIMENSION(:) :: convLayers
    TYPE(maxpoolLayer), ALLOCATABLE, DIMENSION(:) :: maxpoolLayers
    TYPE(avgpoolLayer), ALLOCATABLE, DIMENSION(:) :: avgpoolLayers
    TYPE(addLayer), ALLOCATABLE, DIMENSION(:) :: addLayers
    TYPE(reshapeLayer), ALLOCATABLE, DIMENSION(:) :: reshapeLayers
    CHARACTER(len = 20) :: activation_func
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: weights
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: midWeights
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: largeWeights
    INTEGER :: w_dim1
    INTEGER :: w_dim2
    INTEGER :: w_dim3
    INTEGER :: w_dim4

    REAL (c_double), ALLOCATABLE, DIMENSION(:) :: biases

    ! INTEGER :: activation_func

    CHARACTER(LEN = 100) :: layerName

    INTEGER :: numLayers
    INTEGER :: i
    INTEGER :: readOrNot

    contains

    function c_to_f_string(s) result(str)
        character(kind=c_char, len=1), intent(in) :: s(*)
        character(len=:), allocatable :: str
        integer :: i, n
        n = 0
        do while (n < 4096)
            if (s(n+1) == c_null_char) exit
            n = n + 1
        end do
        allocate(character(len=n) :: str)
        do i = 1, n
            str(i:i) = s(i)
        end do
    end function

    subroutine initialize(model_file, weights_file) bind(c,name="initialize")
        character(kind=c_char, len=1), intent(in), optional :: model_file(*)
        character(kind=c_char, len=1), intent(in), optional :: weights_file(*)
        INTEGER :: Reason, ios
        INTEGER :: modelUnit, weightsUnit
        character(len=:), allocatable :: mpath, wpath

        mpath = "onnxModel.txt"
        wpath = "onnxWeights.txt"
        if (present(model_file))   mpath = c_to_f_string(model_file)
        if (present(weights_file)) wpath = c_to_f_string(weights_file)

        ALLOCATE(lstmLayers(0))
        ALLOCATE(linLayers(0))
        ALLOCATE(convLayers(0))
        ALLOCATE(maxpoolLayers(0))
        ALLOCATE(avgpoolLayers(0))
        ALLOCATE(addLayers(0))
        ALLOCATE(reshapeLayers(0))

        open(newunit=modelUnit, file=mpath, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(error_unit,'(a)') "roseNNa: cannot open model file '"//mpath//"'"
            flush(error_unit)
            error stop 1
        end if
        open(newunit=weightsUnit, file=wpath, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(error_unit,'(a)') "roseNNa: cannot open weights file '"//wpath//"'"
            flush(error_unit)
            error stop 1
        end if

        read(modelUnit, *, iostat=ios) numLayers
        if (ios /= 0) then
            write(error_unit,'(a)') "roseNNa: '"//mpath//"' is empty or malformed"
            flush(error_unit)
            error stop 1
        end if

        readloop: DO i = 1, numLayers
            read(modelUnit, *, IOSTAT=Reason) layerName
            if (Reason < 0) then
                exit readloop
            end if
            if (layerName .eq.  "LSTM") then
                read(modelUnit,*) readOrNot
                CALL read_lstm(modelUnit, weightsUnit, readOrNot)
            else if (layerName .eq. "Gemm") then
                CALL read_linear(modelUnit, weightsUnit)
            else if (layerName .eq. "Conv") then
                CALL read_conv(modelUnit, weightsUnit)
            else if (layerName .eq. "MaxPool") then
                CALL read_maxpool(modelUnit, weightsUnit)
            else if (layerName .eq. "AveragePool") then
                CALL read_avgpool(modelUnit, weightsUnit)
            else if (layerName .eq. "Add") then
                CALL read_add(modelUnit, weightsUnit)
            else if (layerName .eq. "MatMul") then
                cycle
            else if (layerName .eq. "Reshape") then
                read(modelUnit, *) readOrNot
                if (readOrNot .eq. 2) then
                    CALL read_reshape2d(modelUnit, weightsUnit)
                else if (readOrNot .eq. 3) then
                    CALL read_reshape3d(modelUnit, weightsUnit)
                else if (readOrNot .eq. 4) then
                    CALL read_reshape4d(modelUnit, weightsUnit)
                endif
            else if (layerName .eq. "Transpose") then
                cycle
            else if (layerName .eq. "Squeeze") then
                cycle
            else if (layerName .eq. "Pad") then
                cycle
            else if (layerName .eq. "Relu") then
                cycle
            else
                cycle
            end if



        END DO readloop

        close(modelUnit)
        close(weightsUnit)
    end subroutine

    subroutine read_reshape2d(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(reshapeLayer), ALLOCATABLE, DIMENSION(:) :: reshape
        ALLOCATE(reshape(1))
        read(file1, *) w_dim1, w_dim2
        ALLOCATE(weights(w_dim1, w_dim2))
        read(file2, *) weights
        reshape(1)%reshape2d = weights
        DEALLOCATE(weights)
        reshapeLayers = [reshapeLayers, reshape]
        DEALLOCATE(reshape)
    end subroutine

    subroutine read_reshape3d(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(reshapeLayer), ALLOCATABLE, DIMENSION(:) :: reshape
        ALLOCATE(reshape(1))
        read(file1, *) w_dim1, w_dim2, w_dim3
        ALLOCATE(midWeights(w_dim1, w_dim2, w_dim3))
        read(file2, *) midWeights
        reshape(1)%reshape3d = midWeights
        DEALLOCATE(midWeights)
        reshapeLayers = [reshapeLayers, reshape]
        DEALLOCATE(reshape)
    end subroutine

    subroutine read_reshape4d(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(reshapeLayer), ALLOCATABLE, DIMENSION(:) :: reshape
        ALLOCATE(reshape(1))
        read(file1, *) w_dim1, w_dim2, w_dim3, w_dim4
        ALLOCATE(largeWeights(w_dim1, w_dim2, w_dim3, w_dim4))
        read(file2, *) largeWeights
        reshape(1)%reshape4d = largeWeights
        DEALLOCATE(largeWeights)
        reshapeLayers = [reshapeLayers, reshape]
        DEALLOCATE(reshape)
    end subroutine

    subroutine read_add(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(addLayer), ALLOCATABLE, DIMENSION(:) :: add
        ALLOCATE(add(1))
        read(file1, *) w_dim1, w_dim2, w_dim3, w_dim4
        ALLOCATE(largeWeights(w_dim1, w_dim2, w_dim3, w_dim4))
        read(file2, *) largeWeights
        add(1)%adder = largeWeights
        DEALLOCATE(largeWeights)
        addLayers = [addLayers, add]
        DEALLOCATE(add)
    end subroutine

    subroutine read_avgpool(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(avgpoolLayer), ALLOCATABLE, DIMENSION(:) :: avgpool
        ALLOCATE(avgpool(1))
        read(file1, *) w_dim1
        avgpool(1)%kernel_size = w_dim1
        avgpoolLayers = [avgpoolLayers, avgpool]
        DEALLOCATE(avgpool)
    end subroutine

    subroutine read_maxpool(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(maxpoolLayer), ALLOCATABLE, DIMENSION(:) :: maxpool
        ALLOCATE(maxpool(1))
        read(file1, *) w_dim1
        maxpool(1)%kernel_size = w_dim1
        maxpoolLayers = [maxpoolLayers, maxpool]
        DEALLOCATE(maxpool)
    end subroutine
    subroutine read_conv(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(convLayer), ALLOCATABLE, DIMENSION(:) :: conv
        ALLOCATE(conv(1))
        read(file1, *) w_dim1, w_dim2, w_dim3, w_dim4
        ALLOCATE(largeWeights(w_dim1, w_dim2, w_dim3, w_dim4))
        read(file2, *) largeWeights
        conv(1)%weights = largeWeights
        DEALLOCATE(largeWeights)



        read(file1, *) w_dim1
        ALLOCATE(biases(w_dim1))
        read(file2, *) biases
        conv(1)%biases = biases
        DEALLOCATE(biases)


        convLayers = [convLayers, conv]

        DEALLOCATE(conv)
    end subroutine

    subroutine read_lstm(file1, file2, readOrNot)
        INTEGER, INTENT(IN) :: readOrNot
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(lstmLayer), ALLOCATABLE, DIMENSION(:) :: lstm
        ALLOCATE(lstm(1))
        read(file1, *) w_dim1, w_dim2, w_dim3
        ALLOCATE(midWeights(w_dim1,w_dim2,w_dim3))
        read(file2, *) midWeights
        lstm(1)%wih = midWeights
        DEALLOCATE(midWeights)

        read(file1, *) w_dim1, w_dim2, w_dim3
        ALLOCATE(midWeights(w_dim1,w_dim2,w_dim3))
        read(file2, *) midWeights
        lstm(1)%whh = midWeights
        DEALLOCATE(midWeights)


        read(file1, *) w_dim1
        ALLOCATE(biases(w_dim1))
        read(file2, *) biases
        lstm(1)%bih = biases
        DEALLOCATE(biases)

        read(file1, *) w_dim1
        ALLOCATE(biases(w_dim1))
        read(file2, *) biases
        lstm(1)%bhh = biases
        DEALLOCATE(biases)

        if (readOrNot .eq. 1) then
            read(file1, *) w_dim1, w_dim2, w_dim3
            ALLOCATE(midWeights(w_dim1,w_dim2,w_dim3))
            read(file2, *) midWeights
            lstm(1)%hid = midWeights
            DEALLOCATE(midWeights)

            read(file1, *) w_dim1, w_dim2, w_dim3
            ALLOCATE(midWeights(w_dim1,w_dim2,w_dim3))
            read(file2, *) midWeights
            lstm(1)%cell = midWeights
            DEALLOCATE(midWeights)
        endif



        lstmLayers = [lstmLayers, lstm]
        DEALLOCATE(lstm)
    end subroutine

    subroutine read_linear(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(linLayer), ALLOCATABLE,DIMENSION(:) :: lin

        ALLOCATE(lin(1))
        read(file1, *) w_dim1, w_dim2
        ALLOCATE(weights(w_dim1,w_dim2))
        read(file2, *) weights

        read(file1, *) w_dim1
        ALLOCATE(biases(w_dim1))
        read(file2, *) biases

        ! read(file1, *) activation_func

        ! if (activation_func .eq. "Relu") then
        !     lin(1)%fn_ptr => relu2d
        ! else if (activation_func .eq. "Sigmoid") then
        !     lin(1)%fn_ptr => sigmoid2d
        ! else if (activation_func .eq. "Tanh") then
        !     lin(1)%fn_ptr => tanhh2d
        ! else
        !     lin(1)%fn_ptr => null()
        ! end if

        lin(1)%weights = weights
        lin(1)%biases = biases

        DEALLOCATE(weights)
        DEALLOCATE(biases)
        linLayers = [linLayers, lin]
        DEALLOCATE(lin)
    end subroutine



end module
