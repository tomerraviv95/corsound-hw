if __name__ == "__main__":
    voice_faces_dataset = VoiceFacesDataset(audio_embeddings, image_embeddings)

    train_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.train_persons_list,
                                                            triplets_n=10000)
    train_dataloader = DataLoader(train_identity_triplet_dataset, batch_size=64, shuffle=True)

    val_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.val_persons_list,
                                                          triplets_n=5000)
    val_dataloader = DataLoader(val_identity_triplet_dataset, batch_size=64, shuffle=True)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    triplet_loss = nn.TripletMarginLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(10):
        train(net, epoch)

    evaluate(net)