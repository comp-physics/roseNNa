module readTester

    USE derived_types
    USE activation_functions

    implicit none


    TYPE(linLayer), ALLOCATABLE, DIMENSION(:) :: linLayers
    TYPE(lstmLayer), ALLOCATABLE, DIMENSION(:) :: lstmLayers
    REAL, ALLOCATABLE, DIMENSION(:,:) :: weights
    INTEGER :: w_dim1
    INTEGER :: w_dim2

    REAL, ALLOCATABLE, DIMENSION(:) :: biases

    INTEGER :: activation_func

    CHARACTER(LEN = 10) :: layerName

    INTEGER :: numLayers
    INTEGER :: i

    contains
    
    subroutine initialize()
        INTEGER :: Reason
        ALLOCATE(lstmLayers(0))
        ALLOCATE(linLayers(0))
        open(10, file = "model3.txt")
        open(11, file = "weights_biases3.txt")

        read(10, *) numLayers
        
        readloop: DO i = 1, 10
            read(10, *, IOSTAT=Reason) layerName
            if (Reason < 0) then
                exit readloop
            end if
            if (layerName .eq.  "lstm") then
                CALL read_lstm(10, 11)
            else if (layerName .eq. "linear") then
                CALL read_linear(10, 11)
            end if

            
        END DO readloop

    end subroutine
    subroutine read_lstm(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(lstmLayer), ALLOCATABLE, DIMENSION(:) :: lstm
        ALLOCATE(lstm(1))
        read(file1, *) w_dim1, w_dim2
        ALLOCATE(weights(w_dim1,w_dim2))
        read(file2, *) weights
        lstm(1)%wih = weights
        DEALLOCATE(weights)
        
        
        read(file1, *) w_dim1, w_dim2
        ALLOCATE(weights(w_dim1,w_dim2))
        read(file2, *) weights
        lstm(1)%whh = weights
        DEALLOCATE(weights)
        

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

        ALLOCATE(biases(w_dim1))
        read(file2, *) biases

        read(file1, *) activation_func

        if (activation_func == 0) then
            lin(1)%fn_ptr => relu
        else if (activation_func == 1) then
            lin(1)%fn_ptr => sigmoid
        end if

        lin(1)%weights = weights
        lin(1)%biases = biases

        DEALLOCATE(weights)
        DEALLOCATE(biases)
        linLayers = [linLayers, lin]
        DEALLOCATE(lin)
    end subroutine


        
end module

