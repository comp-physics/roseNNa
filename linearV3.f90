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
    
    REAL :: inp1(1,5,1)
    REAL, ALLOCATABLE :: inp3(:,:,:)
    REAL, ALLOCATABLE :: hid1(:,:)
    REAL, ALLOCATABLE :: cell1(:,:)
    REAL, ALLOCATABLE :: out(:,:)
    REAL, ALLOCATABLE :: cellout(:,:)
    REAL :: inp2(2,3,3)
    REAL, ALLOCATABLE :: out2(:,:,:)
    ! REAL, ALLOCATABLE :: out2(:)
    ! REAL, ALLOCATABLE :: cellout2(:)
    ! INTEGER :: i
    REAL :: T1, T2
    ALLOCATE(hid1(2,1))
    ALLOCATE(cell1(2,1))
    inp2 = reshape ((/ 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/), shape(inp2))
    hid1 = reshape ((/ 1.0 , 1.0 /), shape(hid1))
    cell1 = reshape ((/ 1.0 , 1.0/), shape(cell1))
    ALLOCATE(inp3(2,6,6))
    inp3 = RESHAPE((/0.5819, 0.9205, 0.7215, 0.7416, 0.0696, 0.1593, 0.9216, 0.9876, 0.5816,  &
    0.5028, 0.5358, 0.5425, 0.7864, 0.2958, 0.2997, 0.6428, 0.6295, 0.5501, &
    0.4402, 0.1451, 0.2782, 0.0174, 0.4181, 0.2649, 0.8427, 0.5310, 0.1017, &
    0.7868, 0.6185, 0.0911, 0.9202, 0.3077, 0.2393, 0.7275, 0.8895, 0.7733, &
    0.3002, 0.1655, 0.7598, 0.0117, 0.8355, 0.0348, 0.6217, 0.6762, 0.7903, &
    0.4509, 0.3027, 0.7922, 0.2714, 0.2835, 0.5477, 0.6255, 0.6837, 0.1647, &
    0.1745, 0.8284, 0.6694, 0.0722, 0.8537, 0.0729, 0.8240, 0.8679, 0.4999, &
    0.8524, 0.2867, 0.9234, 0.7055, 0.1649, 0.7720, 0.0963, 0.3501, 0.0871/), SHAPE(inp3)) !==testing purposes
    ! out = inp1
    CALL initialize()
    CALL CPU_TIME(T1)
    ! print *, lstmLayers(1)%whh(1,1)
    ! print *, lstmLayers(1)%whh(1,2)


    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
    CALL max_pool(inp3,3,1,out2)
    ! CALL conv(inp2, convLayers(1)%weights, convLayers(1)%biases, 2, 3, 3, 1, out2)
    ! CALL lstm(inp1, hid1, cell1, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh, out, cellout)
    ! CALL linear_layer(out, linLayers(1))
    ! CALL linear_layer(out, linLayers(2))
    ! CALL linear_layer(out, linLayers(3))
    ! CALL linear_layer(out, linLayers(4))

    ! out = linear_layer(out, linLayers(4)%weights, linLayers(4)%biases)
    ! out = linLayers(4)%fn_ptr(out)
    !===================================================================================================================================================
    call CPU_TIME(T2)
    print *, "-------------"
    print *, out2
    ! print *, out2(2,1,1)
    print *, "TIME TAKEN:", T2-T1    

END PROGRAM linear

!=======================================================================================
