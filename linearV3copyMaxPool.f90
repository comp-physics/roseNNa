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

    REAL :: T1, T2
    CALL initialize()
    CALL CPU_TIME(T1)
    
    !========MaxPool Layer============
    CALL max_pool(input,maxpoolLayers(1), 0,     (/0, 0, 0, 0/),     (/3, 3/))

    !===================================================================================================================================================
    call CPU_TIME(T2)
    print *, "-------------"
    print *, input
    print *, "TIME TAKEN:", T2-T1    

END PROGRAM linear

!=======================================================================================
