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
    CALL initialize()
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
    
    SUBROUTINE use_model(input, hidden_state, cell_state, output0, output)
        REAL, ALLOCATABLE, INTENT(IN), DIMENSION(:,:,:) :: input
        REAL, ALLOCATABLE, INTENT(IN), DIMENSION(:,:,:) :: hidden_state
        REAL, ALLOCATABLE, INTENT(IN), DIMENSION(:,:,:) :: cell_state
        REAL, ALLOCATABLE, INTENT(IN), DIMENSION(:,:,:,:) :: output0
        REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:,:) :: output
        REAL :: T1, T2
        CALL CPU_TIME(T1)
        
        !========Transpose============
        input = RESHAPE(input,(/SIZE(input, dim = 2), SIZE(input, dim = 1), SIZE(input, dim = 3)/), order = [2, 1, 3])

        !========LSTM Layer============
        CALL lstm(input, hidden_state, cell_state, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh,&
            & output0)

    end SUBROUTINE
    !===================================================================================================================================================
    call CPU_TIME(T2)
    print *, "-------------"
    output = cell_state
    print *, "TIME TAKEN:", T2-T1    

END PROGRAM linear

!=======================================================================================
