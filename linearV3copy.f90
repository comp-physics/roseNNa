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
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: input
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: hidden_state
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: cell_state
    REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: output0
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: output1
    REAL, ALLOCATABLE, DIMENSION(:,:) :: output2

    REAL :: T1, T2
    ALLOCATE(input(1,2,5))
    ALLOCATE(hidden_state(1,1,2))
    ALLOCATE(cell_state(1,1,2))
    input = RESHAPE((/1,1,1,1,1,1,1,1,1,1/),SHAPE(input))
    hidden_state = RESHAPE((/1,1/),SHAPE(hidden_state))
    cell_state = RESHAPE((/1,1/),SHAPE(cell_state))
    CALL initialize()
    CALL CPU_TIME(T1)
    
    !========Transpose============
    input = RESHAPE(input,(/SIZE(input, dim = 2), SIZE(input, dim = 1), SIZE(input, dim = 3)/), order = [2, 1, 3])

    !========LSTM Layer============
    CALL lstm(input, hidden_state, cell_state, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh, output0)
    print *, output0(1,1,1,:)
    !========Squeeze============
    output1 = RESHAPE(output0,(/SIZE(output0, dim = 1), SIZE(output0, dim = 3), SIZE(output0, dim = 4)/))

    !========Transpose============
    output1 = RESHAPE(output1,(/SIZE(output1, dim = 2), SIZE(output1, dim = 1), SIZE(output1, dim = 3)/), order = [2, 1, 3])

    !========Reshape============
    output1 = RESHAPE(output1,(/SIZE(output1, dim = 3), SIZE(output1, dim = 2), SIZE(output1, dim = 1)/), order = [3,  2,  1 ])
    output2 = RESHAPE(output1,(/2, 2/), order = [2,  1 ])
    
    !========Gemm Layer============
    CALL linear_layer(output2, linLayers(1))

    !========Gemm Layer============
    CALL linear_layer(output2, linLayers(2))
    
    !========Gemm Layer============
    CALL linear_layer(output2, linLayers(3))

    !========Gemm Layer============
    CALL linear_layer(output2, linLayers(4))

    !===================================================================================================================================================
    call CPU_TIME(T2)
    print *, "-------------"
    print *, output2
    print *, "TIME TAKEN:", T2-T1    

END PROGRAM linear

!=======================================================================================
