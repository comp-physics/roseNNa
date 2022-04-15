PROGRAM linear
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE readTester
    ! ===============================================================

    
    IMPLICIT NONE
    
    REAL :: inp1(2,5,1)
    REAL, ALLOCATABLE :: hid1(:,:)
    REAL, ALLOCATABLE :: cell1(:,:)
    REAL, ALLOCATABLE :: out(:,:)
    REAL, ALLOCATABLE :: cellout(:,:)
    ! REAL, ALLOCATABLE :: out2(:)
    ! REAL, ALLOCATABLE :: cellout2(:)
    ! INTEGER :: i
    REAL :: T1, T2
    ALLOCATE(hid1(2,1))
    ALLOCATE(cell1(2,1))
    inp1 = reshape ((/ 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0/), shape(inp1))
    hid1 = reshape ((/ 1.0 , 1.0 /), shape(hid1))
    cell1 = reshape ((/ 1.0 , 1.0/), shape(cell1))

    
    ! out = inp1
    CALL initialize()
    CALL CPU_TIME(T1)
    ! print *, lstmLayers(1)%whh(1,1)
    ! print *, lstmLayers(1)%whh(1,2)


    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
    CALL lstm(inp1, hid1, cell1, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh, out, cellout)
    ! print *, out
    ! CALL lstm_cell(inp1, out2, cellout2, lstmLayers(2)%whh, lstmLayers(2)%wih,x lstmLayers(2)%bih, lstmLayers(2)%bhh, out, cellout)
    ! out = linear_layer(out, linLayers(1)%weights, linLayers(1)%biases)
    ! out = linLayers(1)%fn_ptr(out)

    ! out = linear_layer(out, linLayers(2)%weights, linLayers(2)%biases)
    ! out = linLayers(2)%fn_ptr(out)

    ! out = linear_layer(out, linLayers(3)%weights, linLayers(3)%biases)
    ! out = linLayers(3)%fn_ptr(out)

    ! out = linear_layer(out, linLayers(4)%weights, linLayers(4)%biases)
    ! out = linLayers(4)%fn_ptr(out)
    !===================================================================================================================================================
    call CPU_TIME(T2)
    print *, "-------------"
    print *, out(:, 1)
    ! print *, cellout
    

    print *, "TIME TAKEN:", T2-T1
    ! DEALLOCATE(out)
    ! DEALLOCATE(cellout)    

END PROGRAM linear

!=======================================================================================
