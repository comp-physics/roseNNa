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


    ! INTERFACE
    !     FUNCTION linear_layer(inp, weights, bias) result(output)
    !         real :: output(size(weights, dim=1))
    !         real, intent(in) :: weights(:,:)
    !         real, intent(in) :: inp(:)
    !         real, intent(in) :: bias(:)
    !     END FUNCTION linear_layer
    ! END INTERFACE
    
    REAL :: inp1(5)
    REAL :: hid1(2)
    REAL :: cell1(2)
    REAL, ALLOCATABLE :: out(:)
    REAL, ALLOCATABLE :: cellout(:)
    ! INTEGER :: i
    REAL :: T1, T2
    inp1 = reshape ((/ 1.0 , 1.0, 1.0, 1.0, 1.0  /), shape(inp1))
    hid1 = reshape ((/ 1.0 , 1.0 /), shape(hid1))
    cell1 = reshape ((/ 1.0 , 1.0 /), shape(cell1))

    
    
    CALL initialize()
    CALL CPU_TIME(T1)
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
    CALL lstm_cell(inp1, hid1, cell1, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh, out, cellout)

    out = linear_layer(out, linLayers(1)%weights, linLayers(1)%biases)
    out = linLayers(1)%fn_ptr(out)

    out = linear_layer(out, linLayers(2)%weights, linLayers(2)%biases)
    out = linLayers(2)%fn_ptr(out)

    out = linear_layer(out, linLayers(3)%weights, linLayers(3)%biases)
    out = linLayers(3)%fn_ptr(out)

    out = linear_layer(out, linLayers(4)%weights, linLayers(4)%biases)
    out = linLayers(4)%fn_ptr(out)
    !====================================================================================================================================================
    call CPU_TIME(T2)
    print *, out
    

    print *, "TIME TAKEN:", T2-T1
    DEALLOCATE(out)
    DEALLOCATE(cellout)


    ! out = inp
    ! DO i = 1, SIZE(layers)
    !     out = linear_layer(out, layers(i)%weights, layers(i)%biases)
    !     out = layers(i)%fn_ptr(out)
    ! END DO
    
    
    

END PROGRAM linear

!=======================================================================================
